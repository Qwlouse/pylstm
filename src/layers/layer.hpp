#pragma once
#include "Core.h"
#include <map>
#include <string>
#include <cxxabi.h>

class ViewContainer {
public:
	virtual ~ViewContainer() {}

	Matrix notFound;


	bool contains(const std::string& name) {
		return views.count(name) >= 1;
	}

	Matrix& operator[](const std::string& name) {
		if (contains(name)) {
			return *views[name];
		}
		else {
			return notFound;
		}
	}

	std::vector<std::string> get_view_names() {
	    std::vector<std::string> view_names;
	    for(std::map<std::string,Matrix*>::iterator iter = views.begin(); iter != views.end(); ++iter)
		{
		    view_names.push_back(iter->first);
		}
		return view_names;
	}

	size_t get_size() {
	    return size;
	}

	std::string get_typename() {
	    int status;
        char* demangled = abi::__cxa_demangle(typeid(*this).name(),0,0,&status);
	    return std::string(demangled);
	}

protected:
	void add_view(const std::string& name, Matrix* view) {
		views[name] = view;
	}

	void lay_out(Matrix& buffer) {
		size_t offset = 0;
		for(std::map<std::string,Matrix*>::iterator iter = views.begin(); iter != views.end(); ++iter)
		{
			Matrix* k =  iter->second;
			size_t rows = k->n_rows;
			size_t cols = k->n_columns;
			size_t slices = k->n_slices;
			*k = buffer.subslice(offset, rows, cols, slices);
			offset += k->size;
			ASSERT(offset <= buffer.size);
		}
		size = offset;
	}


private:
	std::map<std::string, Matrix*> views;
	size_t size;
};


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
	virtual ViewContainer* create_weights_view(Matrix& w) = 0;
	virtual ViewContainer* create_fwd_state_view(Matrix& b, size_t n_batches, size_t n_slices) = 0;
	virtual ViewContainer* create_bwd_state_view(Matrix& b, size_t n_batches, size_t n_slices) = 0;
	virtual void forward_pass(ViewContainer& w, ViewContainer& b, Matrix& x, Matrix& y) = 0;
	virtual void backward_pass(ViewContainer& w, ViewContainer& b, ViewContainer& d, Matrix& y, Matrix& in_deltas, Matrix& out_deltas) = 0;
	virtual void gradient(ViewContainer& w, ViewContainer& grad, ViewContainer& b, ViewContainer& d, Matrix& y, Matrix& x, Matrix& out_deltas) = 0;
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
		return L::Weights::estimate_size(in_size, out_size);
	}

	size_t get_fwd_state_size(size_t n_batches, size_t n_slices) {
		return L::FwdState::estimate_size(in_size, out_size, n_batches, n_slices);
	}

	size_t get_bwd_state_size(size_t n_batches, size_t n_slices) {
		return L::BwdState::estimate_size(in_size, out_size, n_batches, n_slices);
	}

	ViewContainer* create_weights_view(Matrix& w) {
		return new typename L::Weights(in_size, out_size, w);
	}

	ViewContainer* create_fwd_state_view(Matrix& b, size_t n_batches, size_t n_slices) {
		return new typename L::FwdState(in_size, out_size, n_batches, n_slices, b);
	}

	ViewContainer* create_bwd_state_view(Matrix& b, size_t n_batches, size_t n_slices) {
		return new typename L::BwdState(in_size, out_size, n_batches, n_slices, b);
	}

	void forward_pass(ViewContainer& w, ViewContainer& b, Matrix& x, Matrix& y) {
		layer.forward(
				dynamic_cast<typename L::Weights&>(w),
				dynamic_cast<typename L::FwdState&>(b),
				x, y);
	}

	void backward_pass(ViewContainer& w, ViewContainer& b, ViewContainer& d, Matrix& y, Matrix& in_deltas, Matrix& out_deltas) {
		layer.backward(
				dynamic_cast<typename L::Weights&>(w),
				dynamic_cast<typename L::FwdState&>(b),
				dynamic_cast<typename L::BwdState&>(d),
				y, in_deltas, out_deltas);
	}

	void gradient(ViewContainer& w, ViewContainer& grad, ViewContainer& b, ViewContainer& d, Matrix& y, Matrix& x, Matrix& out_deltas) {
		layer.gradient(
				dynamic_cast<typename L::Weights&>(w),
				dynamic_cast<typename L::Weights&>(grad),
				dynamic_cast<typename L::FwdState&>(b),
				dynamic_cast<typename L::BwdState&>(d),
				y, x, out_deltas);
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

		size_t fwd_state_size = get_fwd_state_size(n_batches, n_slices);
		Matrix b(1, 1, fwd_state_size);
		ViewContainer* fwd_state = create_fwd_state_view(b, n_batches, n_slices);

		size_t weight_size = get_weight_size();
		ASSERT(w.size == weight_size);
		ViewContainer* weights = create_weights_view(w);

		Matrix y(out_size, n_batches, n_slices);

		forward_pass(*weights, *fwd_state, x, y);
		delete fwd_state;
		delete weights;
		return y;
	}
};

