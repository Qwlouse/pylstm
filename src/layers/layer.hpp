#pragma once
#include <string>
#include <cxxabi.h>

#include "Core.h"
#include "matrix/matrix.h"
#include "view_container.h"


class BaseLayer {
public:
    size_t in_size;
    size_t out_size;
    BaseLayer(size_t in_size, size_t out_size) :
            in_size(in_size),
            out_size(out_size)
        {}

    virtual ~BaseLayer() {};
    virtual std::string get_typename() = 0;
    virtual size_t get_weight_size() = 0;
    virtual size_t get_fwd_state_size(size_t n_batches, size_t n_slices) = 0;
    virtual size_t get_bwd_state_size(size_t n_batches, size_t n_slices) = 0;
    virtual ViewContainer* create_parameter_view(Matrix& w) = 0;
    virtual ViewContainer* create_fwd_state_view(Matrix& b, size_t n_batches, size_t n_slices) = 0;
    virtual ViewContainer* create_bwd_state_view(Matrix& b, size_t n_batches, size_t n_slices) = 0;
    virtual void forward_pass(ViewContainer& w, ViewContainer& b, Matrix& x, Matrix& y) = 0;
    virtual void backward_pass(ViewContainer& w, ViewContainer& b, ViewContainer& d, Matrix& y, Matrix& in_deltas, Matrix& out_deltas) = 0;
    virtual void gradient(ViewContainer& w, ViewContainer& grad, ViewContainer& b, ViewContainer& d, Matrix& y, Matrix& x, Matrix& out_deltas) = 0;
    virtual void Rpass(ViewContainer &w, ViewContainer &v,  ViewContainer &b, ViewContainer &Rb, Matrix &x, Matrix &y, Matrix& Rx, Matrix &Ry) = 0;
    virtual void Rbackward(ViewContainer &w, ViewContainer &b, ViewContainer &d, Matrix &in_deltas, Matrix &out_deltas, ViewContainer &Rb, double lambda, double mu) = 0;
};


template<typename L>
class Layer : public BaseLayer {
public:
    L layer;

    Layer(size_t in_size, size_t out_size) :
        BaseLayer(in_size, out_size),
        layer()
    {}

    Layer(size_t in_size, size_t out_size, L layer) :
        BaseLayer(in_size, out_size),
        layer(layer)
    {}

    size_t get_weight_size() {

        return typename L::Parameters(in_size, out_size).get_size();
    }

    size_t get_fwd_state_size(size_t n_batches, size_t n_slices) {
        return typename L::FwdState(in_size, out_size, n_batches, n_slices).get_size();
    }

    size_t get_bwd_state_size(size_t n_batches, size_t n_slices) {
        return typename L::BwdState(in_size, out_size, n_batches, n_slices).get_size();
    }

    ViewContainer* create_parameter_view(Matrix& w) {
        typename L::Parameters* W = new typename L::Parameters(in_size, out_size);
        W->lay_out(w);
        return W;
    }

    ViewContainer* create_empty_parameter_view() {
        typename L::Parameters* W = new typename L::Parameters(in_size, out_size);
        Matrix w(1, 1, W->get_size());
        W->lay_out(w);
        return W;
    }

    ViewContainer* create_fwd_state_view(Matrix& b, size_t n_batches, size_t n_slices) {
        typename L::FwdState* B = new typename L::FwdState(in_size, out_size, n_batches, n_slices);
        B->lay_out(b);
        return B;
    }

    ViewContainer* create_empty_fwd_state_view(size_t n_batches, size_t n_slices) {
        typename L::FwdState* B = new typename L::FwdState(in_size, out_size, n_batches, n_slices);
        Matrix b(1, 1, B->get_size());
        B->lay_out(b);
        return B;
    }

    ViewContainer* create_bwd_state_view(Matrix& d, size_t n_batches, size_t n_slices) {
        typename L::BwdState* D = new typename L::BwdState(in_size, out_size, n_batches, n_slices);
        D->lay_out(d);
        return D;
    }

    ViewContainer* create_empty_bwd_state_view(size_t n_batches, size_t n_slices) {
        typename L::BwdState* D = new typename L::BwdState(in_size, out_size, n_batches, n_slices);
        Matrix d(1, 1, D->get_size());
        D->lay_out(d);
        return D;
    }

    Matrix create_empty_out_view(size_t n_batches, size_t n_slices) {
        return Matrix(out_size, n_batches, n_slices);
    }

    Matrix create_empty_in_view(size_t n_batches, size_t n_slices) {
        return Matrix(out_size, n_batches, n_slices);
    }

    void forward_pass(ViewContainer& w, ViewContainer& b, Matrix& x, Matrix& y) {
        layer.forward(
                dynamic_cast<typename L::Parameters&>(w),
                dynamic_cast<typename L::FwdState&>(b),
                x, y);
    }

    void backward_pass(ViewContainer& w, ViewContainer& b, ViewContainer& d, Matrix& y, Matrix& in_deltas, Matrix& out_deltas) {
        layer.backward(
                dynamic_cast<typename L::Parameters&>(w),
                dynamic_cast<typename L::FwdState&>(b),
                dynamic_cast<typename L::BwdState&>(d),
                y, in_deltas, out_deltas);
    }

    void gradient(ViewContainer& w, ViewContainer& grad, ViewContainer& b, ViewContainer& d, Matrix& y, Matrix& x, Matrix& out_deltas) {
        layer.gradient(
                dynamic_cast<typename L::Parameters&>(w),
                dynamic_cast<typename L::Parameters&>(grad),
                dynamic_cast<typename L::FwdState&>(b),
                dynamic_cast<typename L::BwdState&>(d),
                y, x, out_deltas);
    }

    void Rpass(ViewContainer &w, ViewContainer &v,  ViewContainer &b, ViewContainer &Rb, Matrix &x, Matrix &y, Matrix& Rx, Matrix &Ry) {
        layer.Rpass(
                dynamic_cast<typename L::Parameters&>(w),
                dynamic_cast<typename L::Parameters&>(v),
                dynamic_cast<typename L::FwdState&>(b),
                dynamic_cast<typename L::FwdState&>(Rb),
                x, y, Rx, Ry);
    }

    void Rbackward(ViewContainer &w, ViewContainer &b, ViewContainer &d, Matrix &in_deltas, Matrix &out_deltas, ViewContainer &Rb, double lambda, double mu) {
        layer.Rbackward(
                dynamic_cast<typename L::Parameters&>(w),
                dynamic_cast<typename L::FwdState&>(b),
                dynamic_cast<typename L::BwdState&>(d),
                in_deltas, out_deltas,
                dynamic_cast<typename L::FwdState&>(Rb),
                lambda, mu);
    }

    std::string get_typename() {
            int status;
            char* demangled = abi::__cxa_demangle(typeid(layer).name(),0,0,&status);
            return std::string(demangled);
        }


    Matrix auto_forward_pass(Matrix& w, Matrix& x) {
        ASSERT(x.n_rows == in_size);
        size_t n_batches = x.n_columns;
        size_t n_slices = x.n_slices;

        ViewContainer* fwd_state = create_empty_fwd_state_view(n_batches, n_slices);

        size_t weight_size = get_weight_size();
        ASSERT(w.size == weight_size);
        ViewContainer* parameters = create_parameter_view(w);

        Matrix y = create_empty_out_view(n_batches, n_slices);

        forward_pass(*parameters, *fwd_state, x, y);
        delete fwd_state;
        delete parameters;
        return y;
    }
};

