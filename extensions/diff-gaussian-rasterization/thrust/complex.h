// Minimal thrust::complex stub for ROCm builds without full thrust headers
// This provides the interface needed by PyTorch's c10::complex

#pragma once

#include <complex>
#include <cmath>

namespace thrust {

template <typename T>
class complex {
public:
    using value_type = T;

    __host__ __device__ complex() : re_(T(0)), im_(T(0)) {}
    __host__ __device__ complex(T re) : re_(re), im_(T(0)) {}
    __host__ __device__ complex(T re, T im) : re_(re), im_(im) {}
    __host__ __device__ complex(const std::complex<T>& c) : re_(c.real()), im_(c.imag()) {}

    template <typename U>
    __host__ __device__ complex(const complex<U>& c) : re_(static_cast<T>(c.real())), im_(static_cast<T>(c.imag())) {}

    __host__ __device__ T real() const { return re_; }
    __host__ __device__ T imag() const { return im_; }
    __host__ __device__ void real(T v) { re_ = v; }
    __host__ __device__ void imag(T v) { im_ = v; }

    __host__ __device__ complex<T>& operator=(T re) {
        re_ = re;
        im_ = T(0);
        return *this;
    }

    template <typename U>
    __host__ __device__ complex<T>& operator=(const complex<U>& c) {
        re_ = static_cast<T>(c.real());
        im_ = static_cast<T>(c.imag());
        return *this;
    }

    __host__ __device__ complex<T>& operator+=(const complex<T>& c) {
        re_ += c.re_;
        im_ += c.im_;
        return *this;
    }

    __host__ __device__ complex<T>& operator-=(const complex<T>& c) {
        re_ -= c.re_;
        im_ -= c.im_;
        return *this;
    }

    __host__ __device__ complex<T>& operator*=(const complex<T>& c) {
        T new_re = re_ * c.re_ - im_ * c.im_;
        T new_im = re_ * c.im_ + im_ * c.re_;
        re_ = new_re;
        im_ = new_im;
        return *this;
    }

    __host__ __device__ complex<T>& operator/=(const complex<T>& c) {
        T denom = c.re_ * c.re_ + c.im_ * c.im_;
        T new_re = (re_ * c.re_ + im_ * c.im_) / denom;
        T new_im = (im_ * c.re_ - re_ * c.im_) / denom;
        re_ = new_re;
        im_ = new_im;
        return *this;
    }

    __host__ __device__ operator std::complex<T>() const {
        return std::complex<T>(re_, im_);
    }

private:
    T re_, im_;
};

// Binary operators
template <typename T>
__host__ __device__ complex<T> operator+(const complex<T>& a, const complex<T>& b) {
    return complex<T>(a.real() + b.real(), a.imag() + b.imag());
}

template <typename T>
__host__ __device__ complex<T> operator-(const complex<T>& a, const complex<T>& b) {
    return complex<T>(a.real() - b.real(), a.imag() - b.imag());
}

template <typename T>
__host__ __device__ complex<T> operator*(const complex<T>& a, const complex<T>& b) {
    return complex<T>(a.real() * b.real() - a.imag() * b.imag(),
                      a.real() * b.imag() + a.imag() * b.real());
}

template <typename T>
__host__ __device__ complex<T> operator/(const complex<T>& a, const complex<T>& b) {
    T denom = b.real() * b.real() + b.imag() * b.imag();
    return complex<T>((a.real() * b.real() + a.imag() * b.imag()) / denom,
                      (a.imag() * b.real() - a.real() * b.imag()) / denom);
}

// Comparison operators
template <typename T>
__host__ __device__ bool operator==(const complex<T>& a, const complex<T>& b) {
    return a.real() == b.real() && a.imag() == b.imag();
}

template <typename T>
__host__ __device__ bool operator!=(const complex<T>& a, const complex<T>& b) {
    return !(a == b);
}

// polar function - required by PyTorch c10::complex
template <typename T>
__host__ __device__ complex<T> polar(const T& r, const T& theta = T()) {
    return complex<T>(r * std::cos(theta), r * std::sin(theta));
}

// abs function
template <typename T>
__host__ __device__ T abs(const complex<T>& c) {
    return std::sqrt(c.real() * c.real() + c.imag() * c.imag());
}

// arg function
template <typename T>
__host__ __device__ T arg(const complex<T>& c) {
    return std::atan2(c.imag(), c.real());
}

// conj function
template <typename T>
__host__ __device__ complex<T> conj(const complex<T>& c) {
    return complex<T>(c.real(), -c.imag());
}

} // namespace thrust
