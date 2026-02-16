#ifndef TENSOR_MAPS_HPP
#define TENSOR_MAPS_HPP

class Tensor {
public:
    int dim;
};

class TensorOps {
  public:
    void flatten(const Tensor &input, Tensor &output);
};

#endif // TENSOR_MAPS_HPP
