## Common network arch

### Preact

`BN` -> `Act` -> `NN`

### MLP

```python
from jamtorch.nn import Seq
network = Seq(d_in).fc(output_size=d_h,bn=True).fc(output_size=d_h, bn=True)
```