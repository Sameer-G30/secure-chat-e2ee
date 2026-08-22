// src/components/Auth/Login.jsx
import { useState } from 'react';

export default function Login({ 
    loginEmail, 
    setLoginEmail, 
    loginPassword, 
    setLoginPassword,
    handleLogin,
    setShowForgotPassword,
    setShowRegister,
    authError,
    authSuccess,
    isLoading
}) {
    const [showPassword, setShowPassword] = useState(false);

    return (
        <div className="form-box">
            <h2>Welcome back</h2>
            <p className="form-subtitle">Sign in to continue</p>
            <form onSubmit={handleLogin}>
                <div className="form-group">
                    <label>Email</label>
                    <input 
                        type="email" 
                        value={loginEmail}
                        onChange={(e) => setLoginEmail(e.target.value)}
                        placeholder="Enter your email"
                        required
                    />
                </div>
                <div className="form-group">
                    <label>Password</label>
                    <div style={{ position: 'relative' }}>
                        <input 
                            type={showPassword ? 'text' : 'password'}
                            value={loginPassword}
                            onChange={(e) => setLoginPassword(e.target.value)}
                            placeholder="Enter your password"
                            required
                            style={{ paddingRight: '48px' }}
                        />
                        <button 
                            type="button"
                            onClick={() => setShowPassword(!showPassword)}
                            style={{
                                position: 'absolute',
                                right: '12px',
                                top: '50%',
                                transform: 'translateY(-50%)',
                                background: 'none',
                                border: 'none',
                                color: 'var(--text-muted)',
                                cursor: 'pointer',
                                fontSize: '14px',
                                padding: '4px',
                            }}
                        >
                            {showPassword ? 'Hide' : 'Show'}
                        </button>
                    </div>
                </div>
                <div className="form-options">
                    <div></div>
                    <a href="#" onClick={(e) => { e.preventDefault(); setShowForgotPassword(true); }}>
                        Forgot password?
                    </a>
                </div>
                {authError && <div style={{ color: '#ef4444', fontSize: '13px', marginBottom: '12px' }}>{authError}</div>}
                {authSuccess && <div style={{ color: '#22c55e', fontSize: '13px', marginBottom: '12px' }}>{authSuccess}</div>}
                <button type="submit" className="btn-primary" disabled={isLoading}>
                    {isLoading ? 'Signing in...' : 'Sign In'}
                </button>
                <p className="form-footer">
                    Don't have an account? <a href="#" onClick={(e) => { e.preventDefault(); setShowRegister(true); }}>Create one</a>
                </p>
            </form>
        </div>
    );
}