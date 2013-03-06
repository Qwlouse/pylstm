#pragma once
#include "layer.hpp"
#include "fwd_layer.h"


BaseLayer* createLayer(const std::string name, size_t in_size, size_t out_size) {
	if (name == "RegularLayer") {
		return new Layer<RegularLayer>(in_size, out_size);
	}
	else {
		return NULL;
	}
}
