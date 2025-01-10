#ifndef ARRAY_CONVERTERS_HPP
#define ARRAY_CONVERTERS_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <armadillo>
#include <cstring> // for std::memcpy

namespace py = pybind11;

// Convert a Python array to an arma::vec
inline arma::vec pyarray_to_arma_vec(const py::array_t<double> &arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim == 1) {
        arma::vec v((double*)buf.ptr, (size_t)buf.shape[0], false, false);
        return v;
    } else if (buf.ndim == 2) {
        size_t length = (size_t)(buf.shape[0] * buf.shape[1]);
        arma::vec v((double*)buf.ptr, length, false, false);
        return v;
    } else {
        throw std::runtime_error("Expected a 1D or 2D vector.");
    }
}

// Convert a Python array to an arma::mat
inline arma::mat pyarray_to_arma_mat(const py::array_t<double> &arr) {
    py::buffer_info buf = arr.request();
    // Interpret the data as (d x n) column-major, then transpose
    arma::mat M((double*)buf.ptr, buf.shape[1], buf.shape[0], false, false);
    return M.t(); // Now M is n x d, matching the Python layout
}


// Convert an arma::vec to a Python array
inline py::array_t<double> arma_vec_to_pyarray(const arma::vec &v) {
    py::array_t<double> result(v.n_elem);
    py::buffer_info buf = result.request();
    std::memcpy(buf.ptr, v.memptr(), v.n_elem * sizeof(double));
    return result;
}

// Convert an arma::mat to a Python array
inline py::array_t<double> arma_mat_to_pyarray(const arma::mat &M) {
    py::array_t<double> result({(py::ssize_t)M.n_rows, (py::ssize_t)M.n_cols});
    py::buffer_info buf = result.request();
    std::memcpy(buf.ptr, M.memptr(), M.n_elem * sizeof(double));
    return result;
}

inline arma::mat Moebius_S(const arma::mat &X, arma::vec mu, double rho){
  
  arma::mat Y = (1-rho*rho)*(X.each_row() + rho*mu.t());
  Y = Y.each_col()/(1+2*rho*X*mu+rho*rho);
  Y = Y.each_row() + rho*mu.t();
  
  return Y;
}

#endif // ARRAY_CONVERTERS_HPP
