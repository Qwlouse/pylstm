#include "matrix_container.h"

#include <cxxabi.h>
#include "Core.h"

bool MatrixContainer::contains(const std::string& name) {
    return views.count(name) >= 1;
}

Matrix& MatrixContainer::operator[](const std::string& name) {
    if (contains(name)) {
        return *views[name];
    }
    else {
        return notFound;
    }
}

std::vector<std::string> MatrixContainer::get_view_names() {
    std::vector<std::string> view_names;
    for(std::map<std::string,Matrix*>::iterator iter = views.begin(); iter != views.end(); ++iter)
    {
        view_names.push_back(iter->first);
    }
    return view_names;
}

size_t MatrixContainer::get_size() {
    return size;
}

std::string MatrixContainer::get_typename() {
    int status;
    char* demangled = abi::__cxa_demangle(typeid(*this).name(),0,0,&status);
    return std::string(demangled);
}

void MatrixContainer::lay_out(Matrix& buffer) {
    size_t offset = 0;
    for(std::map<std::string,Matrix*>::iterator iter = views.begin(); iter != views.end(); ++iter)
    {
        Matrix* k =  iter->second;
        size_t rows = k->n_rows;
        size_t cols = k->n_columns;
        size_t slices = k->n_slices;
        *k = buffer.sub_matrix(offset, rows, cols, slices);
        offset += k->size;
        ASSERT(offset <= buffer.size);
    }
    size = offset;
}

void MatrixContainer::add_view(const std::string& name, Matrix* view) {
    views[name] = view;
    size += view->size;
}
