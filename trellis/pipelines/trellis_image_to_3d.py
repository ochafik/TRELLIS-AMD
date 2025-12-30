from typing import *
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import rembg
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp


class TrellisImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisImageTo3DPipeline, TrellisImageTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline
    
    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        import os

        # AMD ROCm: Disable xformers for DINOv2 (not compatible with AMD)
        # This forces DINOv2 to use standard PyTorch attention
        _is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        if _is_rocm:
            os.environ['XFORMERS_DISABLED'] = '1'

        dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        dinov2_model.eval()

        # AMD ROCm: Run DINOv2 on CPU to avoid hipBLAS/Tensile bugs on gfx1151
        # The Tensile GEMM kernels have issues on Strix Point (gfx1151) that cause
        # incorrect output or crashes. CPU is slower but produces correct results.
        if _is_rocm:
            print("[AMD] Running DINOv2 on CPU to avoid hipBLAS/Tensile bugs")
            print("[AMD] This is slower but produces correct conditioning output")
            dinov2_model.float().cpu()  # Keep on CPU
            self._dinov2_on_cpu = True
        else:
            self._dinov2_on_cpu = False

        self.models['image_cond_model'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform

    def cuda(self) -> None:
        """
        Move models to CUDA, but keep DINOv2 on CPU for AMD ROCm.
        """
        super().cuda()
        # On AMD ROCm, move DINOv2 back to CPU after parent moves everything to CUDA
        if getattr(self, '_dinov2_on_cpu', False):
            self.models['image_cond_model'].cpu()

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            output = rembg.remove(input, session=self.rembg_session)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        import os
        _debug_encode = os.environ.get('DEBUG_ENCODE', '0') == '1'

        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            if _debug_encode:
                print(f"[DEBUG_ENCODE] Input: {len(image)} images")
                for idx, img in enumerate(image):
                    arr = np.array(img)
                    print(f"  Image {idx}: size={img.size}, mode={img.mode}, np_min={arr.min()}, np_max={arr.max()}, np_mean={arr.mean():.2f}")
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            if _debug_encode:
                for idx, arr in enumerate(image):
                    print(f"  After RGB conv {idx}: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
            if _debug_encode:
                print(f"  Stacked tensor: shape={image.shape}, device={image.device}")
                for idx in range(image.shape[0]):
                    print(f"    Tensor {idx}: min={image[idx].min():.4f}, max={image[idx].max():.4f}, mean={image[idx].mean():.4f}")
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        image = self.image_cond_model_transform(image).to(self.device)
        if _debug_encode:
            print(f"  After transform: shape={image.shape}")
            for idx in range(image.shape[0]):
                print(f"    Normalized {idx}: min={image[idx].min():.4f}, max={image[idx].max():.4f}, mean={image[idx].mean():.4f}")
            if image.shape[0] > 1:
                diff = (image[0] - image[1]).abs()
                print(f"    Tensor diff (0 vs 1): mean={diff.mean():.4f}, max={diff.max():.4f}")

        # AMD ROCm: Run DINOv2 on CPU to avoid Tensile kernel bugs
        _is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        if _is_rocm and getattr(self, '_dinov2_on_cpu', False):
            if _debug_encode:
                print(f"  [AMD] Running DINOv2 on CPU...")
            # Move input to CPU for DINOv2
            image_cpu = image.cpu()
            features = self.models['image_cond_model'](image_cpu, is_training=True)['x_prenorm']
            # Move output back to GPU
            features = features.to(self.device)
            if _debug_encode:
                print(f"  [AMD] DINOv2 complete, features moved to {self.device}")
        else:
            # Standard GPU path (for NVIDIA or non-AMD systems)
            if hasattr(self.models['image_cond_model'], 'parameters'):
                model_device = next(self.models['image_cond_model'].parameters()).device
                if model_device != image.device:
                    if _debug_encode:
                        print(f"  WARNING: Model on {model_device}, input on {image.device}. Moving model...")
                    self.models['image_cond_model'].to(image.device)
            features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']

        # AMD ROCm sanity check: verify DINOv2 is actually processing input correctly
        # On some AMD setups, DINOv2 may return constant output regardless of input
        if image.shape[0] > 1:
            input_var = (image[0] - image[1]).abs().mean().item()
            output_var = (features[0] - features[1]).abs().mean().item()
            if input_var > 0.01 and output_var < 0.001:
                print(f"[WARNING] DINOv2 may not be encoding images correctly!")
                print(f"  Input images are different (var={input_var:.4f})")
                print(f"  But DINOv2 output is identical (var={output_var:.6f})")
                print(f"  This is a known issue on some AMD/ROCm setups.")

        if _debug_encode:
            print(f"  After DINOv2: shape={features.shape}")
            for idx in range(features.shape[0]):
                print(f"    Features {idx}: min={features[idx].min():.4f}, max={features[idx].max():.4f}, mean={features[idx].mean():.4f}")
            if features.shape[0] > 1:
                diff = (features[0] - features[1]).abs()
                print(f"    Feature diff (0 vs 1): mean={diff.mean():.4f}, max={diff.max():.4f}")

        patchtokens = F.layer_norm(features, features.shape[-1:])
        if _debug_encode:
            print(f"  After LayerNorm: shape={patchtokens.shape}")
            for idx in range(patchtokens.shape[0]):
                print(f"    Patchtokens {idx}: min={patchtokens[idx].min():.4f}, max={patchtokens[idx].max():.4f}")
            if patchtokens.shape[0] > 1:
                diff = (patchtokens[0] - patchtokens[1]).abs()
                print(f"    Patchtokens diff (0 vs 1): mean={diff.mean():.4f}, max={diff.max():.4f}")

        return patchtokens
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        # Warmup torchsparse kernels on AMD ROCm before image encoding
        # This prevents HIP kernel launch failures with torchsparse after DINOv2 runs
        sp.warmup_rocm_sparse()

        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']

        # AMD HIP FIX: Force consistent float32 precision for flow model
        if hasattr(flow_model, 'convert_to_fp32'):
            flow_model.convert_to_fp32()
        flow_model.float()
        if hasattr(flow_model, 'dtype'):
            flow_model.dtype = torch.float32
        if hasattr(flow_model, 'use_fp16'):
            flow_model.use_fp16 = False

        # AMD HIP FIX: Ensure conditioning tensors are float32
        cond_fp32 = {}
        for k, v in cond.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                cond_fp32[k] = v.float()
            else:
                cond_fp32[k] = v

        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).float().to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond_fp32,
            **sampler_params,
            verbose=True
        ).samples
        
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']

        # AMD HIP FIX: Force consistent float32 precision to avoid mixed dtype errors
        # The decoder has mixed precision (some layers fp16, some fp32) which causes
        # HIP kernel launch failures. fp16 also causes rocBLAS/Tensile errors on RDNA.
        # Solution: Use the model's convert_to_fp32() method which sets self.dtype AND converts params.
        if hasattr(decoder, 'convert_to_fp32'):
            decoder.convert_to_fp32()
        else:
            decoder.float()
        z_s = z_s.float()  # Convert input to float32

        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

        return coords

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        # AMD HIP FIX: Ensure all decoders use consistent float32 precision
        for decoder_name in ['slat_decoder_mesh', 'slat_decoder_gs', 'slat_decoder_rf']:
            if decoder_name in self.models:
                decoder = self.models[decoder_name]
                decoder.float()
                if hasattr(decoder, 'dtype'):
                    decoder.dtype = torch.float32
                if hasattr(decoder, 'use_fp16'):
                    decoder.use_fp16 = False

        # Ensure SLAT tensor features are float32
        if slat.feats.dtype != torch.float32:
            slat = slat.replace(slat.feats.float())

        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        if 'radiance_field' in formats:
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']

        # AMD HIP FIX: Force consistent float32 precision to avoid mixed dtype errors
        # convert_to_fp32() only converts some layers; also call float() for everything
        # Also set self.dtype since the forward method uses it for type conversions
        if hasattr(flow_model, 'convert_to_fp32'):
            flow_model.convert_to_fp32()
        flow_model.float()
        flow_model.dtype = torch.float32
        flow_model.use_fp16 = False

        # AMD HIP FIX: Ensure conditioning tensors are also float32
        cond_fp32 = {}
        for k, v in cond.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                cond_fp32[k] = v.float()
            else:
                cond_fp32[k] = v

        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).float().to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond_fp32,
            **sampler_params,
            verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        """
        Inject a sampler with multiple images as condition.
        
        Args:
            sampler_name (str): The name of the sampler to inject.
            num_images (int): The number of images to condition on.
            num_steps (int): The number of steps to run the sampler for.
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f'_old_inference_model', sampler._inference_model)

        if mode == 'stochastic':
            if num_images > num_steps:
                print(f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m")

            cond_indices = (np.arange(num_steps) % num_images).tolist()
            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
        
        elif mode =='multidiffusion':
            from .samplers import FlowEulerSampler
            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    return pred
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        yield

        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f'_old_inference_model')

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ) -> dict:
        """
        Run the pipeline with multiple images as condition

        Args:
            images (List[Image.Image]): The multi-view images of the assets
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        cond = self.get_cond(images)
        cond['neg_cond'] = cond['neg_cond'][:1]
        torch.manual_seed(seed)
        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('sparse_structure_sampler', len(images), ss_steps, mode=mode):
            coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('slat_sampler', len(images), slat_steps, mode=mode):
            slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
