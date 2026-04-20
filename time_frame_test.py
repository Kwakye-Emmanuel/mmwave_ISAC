import numpy as np

# Parameters
M = 8
Kd = 2
Pt = 1
sigma2 = 1
rho = 0.5
num_trials = 5000

# Time parameters
T = 1
tau = 0.01
L = 16
Tc_cao = T - tau * L   # Cao paper
Tc_yours = T           # your assumption

# Steering vector
def steering(theta, M):
    n = np.arange(M)
    return np.exp(1j * np.pi * n * np.sin(theta)) / np.sqrt(M)

# Generate Rayleigh user channel
def user_channel():
    return (np.random.randn(M) + 1j*np.random.randn(M)) / np.sqrt(2)

# Eve channel (LoS)
def eve_channel(theta):
    return steering(theta, M)

# ZF beamforming
def zf(H):
    return H.conj().T @ np.linalg.inv(H @ H.conj().T)

# Null space projector
def null_space(H):
    return np.eye(M) - H.conj().T @ np.linalg.inv(H @ H.conj().T) @ H

# Secrecy computation
def secrecy_rate(sinr_u, sinr_e, Tc):
    return (Tc/T) * np.maximum(np.log2(1+sinr_u) - np.log2(1+sinr_e), 0)

results = {"no_sensing": [], "your": [], "cao": []}

for _ in range(num_trials):
    
    # Random user channels
    H = np.array([user_channel() for _ in range(Kd)])
    
    # Random Eve angle
    theta_e = np.random.uniform(-np.pi/2, np.pi/2)
    g = eve_channel(theta_e)
    
    # ZF beamforming
    W = zf(H)
    W = np.sqrt(rho*Pt/Kd) * W
    
    # Null space
    V = null_space(H)
    
    # ---------- (a) NO SENSING ----------
    R_iso = (1-rho)*Pt/(M-Kd) * V @ V.conj().T
    
    sinr_u = np.abs(H[0].conj() @ W[:,0])**2 / sigma2
    sinr_e = np.abs(g.conj() @ W[:,0])**2 / (g.conj() @ R_iso @ g + sigma2)
    
    results["no_sensing"].append(secrecy_rate(sinr_u, sinr_e, T))
    
    # ---------- (b) YOUR METHOD ----------
    # perfect estimation assumed for simplicity
    u = V.conj().T @ g
    u = u / np.linalg.norm(u)
    R_dir = (1-rho)*Pt * V @ np.outer(u, u.conj()) @ V.conj().T
    
    sinr_e2 = np.abs(g.conj() @ W[:,0])**2 / (g.conj() @ R_dir @ g + sigma2)
    
    results["your"].append(secrecy_rate(sinr_u, sinr_e2, Tc_yours))
    
    # ---------- (c) CAO METHOD ----------
    results["cao"].append(secrecy_rate(sinr_u, sinr_e2, Tc_cao))


# Average results
print("Average Secrecy Rates:")
print("No sensing: ", np.mean(results["no_sensing"]))
print("Your method:", np.mean(results["your"]))
print("Cao method:", np.mean(results["cao"]))