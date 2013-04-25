#pragma once

#include <map>
#include <string>

#include "matrix/matrix.h"


class MatrixContainer {
public:
    MatrixContainer() : size(0) {}

    virtual ~MatrixContainer() {}

    Matrix notFound;

    bool contains(const std::string& name);

    Matrix& operator[](const std::string& name);

    std::vector<std::string> get_view_names();

    size_t get_size();

    std::string get_typename();

    void lay_out(Matrix& buffer);

protected:
    void add_view(const std::string& name, Matrix* view);

private:
    std::map<std::string, Matrix*> views;
    size_t size;
};