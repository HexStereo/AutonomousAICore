# Tensor & Memory Design Specification

## 1. Tensor Memory Layout & Strides
A `Tensor` in AutonomousAICore represents a view over a multi-dimensional row-major or strided array:
$$\text{offset}(\mathbf{i}) = \text{storage\_offset} + \sum_{k=0}^{\text{rank}-1} i_k \cdot \text{stride}_k$$

### Contiguity
A tensor is contiguous if and only if:
$$\text{stride}_{r-1} = 1 \quad \text{and} \quad \text{stride}_k = \text{stride}_{k+1} \cdot \text{dim}_{k+1} \quad \forall k < r-1 \text{ (where } \text{dim}_k > 1\text{)}$$

### Zero-Copy Operations
- `transpose(d0, d1)` swaps $\text{dim}_0 \leftrightarrow \text{dim}_1$ and $\text{stride}_0 \leftrightarrow \text{stride}_1$ without copying data.
- `permute(dims)` reorders the dimension and stride vectors directly.
- `view(target_shape)` creates a new tensor referencing the identical `Storage` if the tensor is contiguous.

## 2. Memory Subsystem Pools
- CPU `HostMemoryPool`: Segregated size classes, 64-byte alignment, zero-overhead allocation recycling.
- GPU `DeviceMemoryPool`: CUDA stream-aware block recycling using CUDA events.
- RAII Sahipliği: Her tensor bir `std::shared_ptr<Storage>` tutar. Son tensör yok edildiğinde blok havuza otomatik olarak iade edilir.
