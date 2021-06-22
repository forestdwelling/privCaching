## Personalized Succinct Histogram
dimension reduction (d->m, d>>n)
$$m=\frac{\ln{(d+1)}\ln{\frac{2}{\beta}}\epsilon^2n}{\ln{\frac{2d}{\beta}}}$$

randomly initiate matrix $\Phi$:
$$\Phi=\{\frac{1}{\sqrt{m}}, -\frac{1}{\sqrt{m}}\}^{m\times d}$$

for each user, randomly select $i\in \{1,...,m\}$, $j\in \{1,...,l\}$:
$$z_i \leftarrow z_i + m\cdot\Phi_{i,data[j]}$$

for each key $c$:
$$freq(i)=\langle\Phi_{\cdot,c}, z\cdot l\rangle$$

## Similarity
$$sim=\sum_{c\in D\cap D'}f_c$$