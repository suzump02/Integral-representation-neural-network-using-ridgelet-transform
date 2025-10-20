# Integral-representation-neural-network-using-ridgelet-transform
Ridgelet変換によるNNsの数値計算

## Theoretical framework
We choose the hyperbolic tangent function $\tanh(z)$ as the active function $\eta(z)$. The Fourier transform of the hyperbolic tangent function is given by:

```math
\widehat{\eta}(\zeta) = \frac{-i\pi}{2 \sinh(\frac{\pi}{2}\zeta)}.
```

See "note/ridgelet_reconstruction_note_v1.pdf" for derivation of this formula.

**REMARK:** Coefficients of the reconduction formula ($m=1$)

```math
K_{\eta, \psi} = \int_{-\infty}^{\infty} \frac{\overline{\widehat{\psi}(\zeta)}{\widehat{\eta}(\zeta)}}{|\zeta|} d\zeta.
```

Then, to confirm the reconstruction theorem, the corresponding　function $\psi$ was set up as follows:

```math
\widehat{\psi}(\zeta) = i^{2k-1}\zeta^{2k-1}e^{-\zeta^2}, \quad k\in\mathbb{N}
```

For this equation, by computing the inverse Fourier transform, we can write $\psi$ as a function of $z$ as follows.

```math
\psi(z) = 2\sqrt{\pi}\cdot(-1)\cdot\left(\frac{1}{2}\right)^{2k-1}\cdot H_{2k-1}\left(\frac{z}{2}\right)\cdot e^{-z^2/4}
```

where, $H_n$ is the Hermite function.




