// src/components/Auth/Register.jsx
import { useState } from 'react';

export default function Register({
    registerName,
    setRegisterName,
    registerEmail,
    setRegisterEmail,
    registerPassword,
    setRegisterPassword,
    registerConfirmPassword,
    setRegisterConfirmPassword,
    handleRegister,
    setShowRegister,
    authError,
    authSuccess,
    isLoading,
    passwordStrength,
    checkPasswordStrength,
    showPassword,
    setShowPassword
}) {
    return (
        <div className="form-box">
            <button className="back-btn" onClick={() => { setShowRegister(false); }}>
                ← Back
            </button>
            <h2>Create Account</h2>
            <p className="form-subtitle">Start your secure messaging journey</p>
            <form onSubmit={handleRegister}>
                <div className="form-group">
                    <label>Full Name</label>
                    <input 
                        type="text" 
                        value={registerName}
                        onChange={(e) => setRegisterName(e.target.value)}
                        placeholder="Enter your full name"
                        required
                    />
                </div>
                <div className="form-group">
                    <label>Email</label>
                    <input 
                        type="email" 
                        value={registerEmail}
                        onChange={(e) => setRegisterEmail(e.target.value)}
                        placeholder="Enter your email"
                        required
                    />
                </div>
                <div className="form-group">
                    <label>Password</label>
                    <div style={{ position: 'relative' }}>
                        <input 
                            type={showPassword ? 'text' : 'password'}
                            value={registerPassword}
                            onChange={(e) => {
                                setRegisterPassword(e.target.value);
                                checkPasswordStrength(e.target.value);
                            }}
                            placeholder="Create a strong password"
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
                    {registerPassword && (
                        <div style={{ marginTop: '8px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                <div style={{ 
                                    flex: 1, 
                                    height: '3px', 
                                    background: 'var(--bg-tertiary)', 
                                    borderRadius: '4px',
                                    overflow: 'hidden',
                                }}>
                                    <div style={{ 
                                        width: `${(passwordStrength.score / 5) * 100}%`, 
                                        height: '100%', 
                                        background: passwordStrength.color,
                                        transition: 'width 0.3s ease',
                                        borderRadius: '4px',
                                    }} />
                                </div>
                                <span style={{ fontSize: '12px', fontWeight: '500', color: passwordStrength.color, minWidth: '50px' }}>
                                    {passwordStrength.label}
                                </span>
                            </div>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: '6px' }}>
                                {[
                                    { key: 'length', label: '8+ chars' },
                                    { key: 'uppercase', label: 'Uppercase' },
                                    { key: 'lowercase', label: 'Lowercase' },
                                    { key: 'number', label: 'Number' },
                                    { key: 'special', label: 'Special' },
                                ].map((req) => (
                                    <span key={req.key} style={{
                                        fontSize: '10px',
                                        padding: '2px 10px',
                                        borderRadius: '12px',
                                        background: passwordStrength.requirements[req.key] ? 'rgba(79, 70, 229, 0.15)' : 'var(--bg-tertiary)',
                                        color: passwordStrength.requirements[req.key] ? 'var(--accent)' : 'var(--text-muted)',
                                        transition: 'all 0.3s ease',
                                    }}>
                                        {passwordStrength.requirements[req.key] ? '✓' : '○'} {req.label}
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
                <div className="form-group">
                    <label>Confirm Password</label>
                    <input 
                        type="password" 
                        value={registerConfirmPassword}
                        onChange={(e) => setRegisterConfirmPassword(e.target.value)}
                        placeholder="Confirm your password"
                        required
                    />
                </div>
                {authError && <div style={{ color: '#ef4444', fontSize: '13px', marginBottom: '12px' }}>{authError}</div>}
                {authSuccess && <div style={{ color: '#22c55e', fontSize: '13px', marginBottom: '12px' }}>{authSuccess}</div>}
                <button type="submit" className="btn-primary" disabled={isLoading}>
                    {isLoading ? 'Creating account...' : 'Create Account'}
                </button>
                <p className="form-footer">
                    Already have an account? <a href="#" onClick={(e) => { e.preventDefault(); setShowRegister(false); }}>Sign In</a>
                </p>
            </form>
        </div>
    );
}