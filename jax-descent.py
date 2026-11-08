import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Vectorized Function & Hardcoded Gradient
# ------------------------------------------------------------------

@jax.jit
def f(x):
    """
    Vectorized scalar function f: R^2 -> R
    x: Array of shape (2,) or (N, 2)
    """
    x1, x2 = x[..., 0], x[..., 1]
    return jnp.sin(x1) * jnp.cos(x2) + jnp.sin(0.5 * x1) * jnp.cos(0.5 * x2)

@jax.jit
def grad_f_hardcoded(x):
    """
    Hardcoded analytical gradient of f(x).
    x: Array of shape (2,)
    Returns: Array of shape (2,)
    """
    x1, x2 = x[0], x[1]
    df_dx1 = jnp.cos(x1) * jnp.cos(x2) + 0.5 * jnp.cos(0.5 * x1) * jnp.cos(0.5 * x2)
    df_dx2 = -jnp.sin(x1) * jnp.sin(x2) - 0.5 * jnp.sin(0.5 * x1) * jnp.sin(0.5 * x2)
    return jnp.array([df_dx1, df_dx2])


# Optional check: Verify analytical gradient matches JAX automatic differentiation
# jax_grad = jax.grad(f)(jnp.array([2.0, 3.0]))
# hardcoded_grad = grad_f_hardcoded(jnp.array([2.0, 3.0]))
# print("AD match check:", jnp.allclose(jax_grad, hardcoded_grad))


# ------------------------------------------------------------------
# 2. Optimization Procedures (Vanilla GD & Momentum GD)
# ------------------------------------------------------------------

def vanilla_gradient_descent(x_init, lr=0.1, n_steps=100):
    """Standard Gradient Descent."""
    history_x = [x_init]
    history_grad_norm = []
    
    x = x_init
    for _ in range(n_steps):
        g = grad_f_hardcoded(x)
        history_grad_norm.append(jnp.linalg.norm(g))
        x = x - lr * g
        history_x.append(x)
        
    return jnp.array(history_x), jnp.array(history_grad_norm)

def momentum_gradient_descent(x_init, lr=0.1, beta=0.9, n_steps=100):
    """Gradient Descent with Polyak Momentum."""
    history_x = [x_init]
    history_grad_norm = []
    
    x = x_init
    v = jnp.zeros_like(x)
    
    for _ in range(n_steps):
        g = grad_f_hardcoded(x)
        history_grad_norm.append(jnp.linalg.norm(g))
        v = beta * v + lr * g
        x = x - v
        history_x.append(x)
        
    return jnp.array(history_x), jnp.array(history_grad_norm)


# ------------------------------------------------------------------
# 3. Execution & Visualization
# ------------------------------------------------------------------

# Configuration
x_start = jnp.array([8.0, 8.0])  # Starting point in [0, 10]^2 domain
n_steps = 80
lr = 0.15
beta = 0.85

# Run optimizers
path_gd, grads_gd = vanilla_gradient_descent(x_start, lr=lr, n_steps=n_steps)
path_mom, grads_mom = momentum_gradient_descent(x_start, lr=lr, beta=beta, n_steps=n_steps)

# Prepare contour mesh in domain [0, 10] x [0, 10]
grid_res = 200
x1_vals = jnp.linspace(0, 10, grid_res)
x2_vals = jnp.linspace(0, 10, grid_res)
X1, X2 = jnp.meshgrid(x1_vals, x2_vals)
grid_pts = jnp.stack([X1, X2], axis=-1)
Z = f(grid_pts)  # Vectorized call over (200, 200, 2) array

# Figure setup
plt.figure(figsize=(14, 6))

# Plot 1: Optimization Paths over Loss Contour
plt.subplot(1, 2, 1)
contour = plt.contourf(X1, X2, Z, levels=30, cmap='viridis', alpha=0.85)
plt.colorbar(contour, label=r'$f(x_1, x_2)$')

plt.plot(path_gd[:, 0], path_gd[:, 1], 'o-', color='crimson', label='Vanilla GD', markersize=3, linewidth=1.5)
plt.plot(path_mom[:, 0], path_mom[:, 1], 'o-', color='cyan', label=f'Momentum ($\\beta={beta}$)', markersize=3, linewidth=1.5)
plt.plot(x_start[0], x_start[1], 'r*', markersize=12, label='Start (8, 8)')

plt.xlim([0, 10])
plt.ylim([0, 10])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title('Optimization Paths on Contour Map $[0, 10]^2$')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

# Plot 2: Gradient Norm Convergence
plt.subplot(1, 2, 2)
plt.plot(grads_gd, color='crimson', label='Vanilla GD', linewidth=2)
plt.plot(grads_mom, color='cyan', label=f'Momentum ($\\beta={beta}$)', linewidth=2)

plt.yscale('log')
plt.xlabel('Iteration Step')
plt.ylabel(r'Gradient Norm $\|\nabla f(\mathbf{x})\|_2$ (Log Scale)')
plt.title('Gradient Norm vs. Iterations')
plt.legend()
plt.grid(True, which="both", linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()