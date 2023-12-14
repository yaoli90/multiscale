# multiscale

* elliptic_1D

  $-\frac{d}{dx}(a(\frac{x}{\epsilon})\frac{d}{dx}u_\epsilon)=f$, $x\in[0,\pi]$,

  $u_\epsilon(0)=u_\epsilon(\pi)=0$

  $a(\frac{x}{\epsilon})=0.5\sin(\frac{2\pi x}{\epsilon})+2$, $\epsilon=\frac{1}{8}$

  | epsilon | PINN | NH-PINN | multiscale-PINN
  | ----------- | ----------- | ----------- |----------- |
  | 1/8 | 0.7515 | - | 0.0068 |

* slowly_varying

   $-\frac{d}{dx}(a(x,\frac{x}{\epsilon})\frac{d}{dx}u_\epsilon)=f$, $x\in[0,\pi]$,

   $u_\epsilon(0)=u_\epsilon(\pi)=0$

   $a(x,\frac{x}{\epsilon})=0.5\sin(\frac{2\pi x}{\epsilon})+\sin(x)+2$, $\epsilon=\frac{1}{8}$, or $\epsilon=\frac{1}{10}$

   | epsilon | PINN | NH-PINN | multiscale-PINN
   | ----------- | ----------- | ----------- |----------- |
   | 1/8 | 0.3504 | 0.00582 | 0.0004 |
   | 1/10 | 0.9877 | - | 0.0006 |


* elliptic_2D

   $-\frac{\partial}{\partial x_i}(a(\frac{x}{\epsilon})\frac{\partial}{\partial x_i}u_\epsilon(x))=f(x)$, $x\in\Omega$,

   $u_\epsilon(x)=0$, $x\in\partial\Omega$, $\Omega\in[0,1]^2$

   $a(\frac{x}{\epsilon})=\sin(\frac{2\pi x_1}{\epsilon})\cos(\frac{2\pi x_2}{\epsilon})+2$, $\epsilon=\frac{1}{8}$

   $f(x)=\sin(x_1)+\cos(x_2)$

   | epsilon | PINN | NH-PINN | multiscale-PINN
   | ----------- | ----------- | ----------- |----------- |
   | 1/8 | 0.7664 | 0.08299 | 0.0178 |

* diff_reac

   $\frac{\partial u_\epsilon}{\partial t}-D\nabla\cdot\nabla u_\epsilon + \frac{1}{\epsilon}r(\frac{x}{\epsilon})u_\epsilon=f$, $x\in\Omega$, $t\in (0,1]$,

   $u_\epsilon(x,t)=0$, $x\in\partial\Omega$, $\Omega=[-\pi, \pi]$,

   $u_\epsilon(x,0)=0.01\sin(x)$,

   $r(\frac{x}{\epsilon}) = \cos(\frac{x}{\epsilon})$, $D=2$, $\epsilon=\frac{1}{10}$ or $\epsilon=\frac{1}{50}$,

   $f(x)=\sin(2\pi x)$

   | epsilon | PINN | NH-PINN | multiscale-PINN
   | ----------- | ----------- | ----------- |----------- |
   | 1/10 | 0.0077 | 0.012388 | 0.0034 |
   | 1/50 | 0.8970 | 0.018582 | 0.0092 |
