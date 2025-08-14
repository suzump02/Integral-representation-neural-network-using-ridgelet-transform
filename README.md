# Integral-representation-neural-network-using-ridgelet-transform
Ridgelet変換によるNNsの数値計算

## Theoretical framework
We choose the hyperbolic tangent function $\tanh(z)$ as the active function $\eta(z)$. The Fourier transform of the hyperbolic tangent function is given by :
$$
\begin{align*}
\widehat{\eta}(\zeta) = \frac{-i\pi}{2 \sinh(\frac{\pi}{2}\zeta)}.
\end{align*}
$$

**REMARK:**Coefficients of the reconduction formula($m=1$)
$$
\begin{align*}
K_{\eta, \psi} = \int_{-\infty}^{\infty} \frac{\overline{\widehat{\psi}(\zeta)}{\widehat{\eta}(\zeta)}}{|\zeta|} d\zeta
\end{align*}.
$$

Then, to confirm the reconstruction theorem, the corresponding　function $\psi$ was set up as follows