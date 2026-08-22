// src/components/Auth/ForgotPassword.jsx
export default function ForgotPassword({
    forgotEmail,
    setForgotEmail,
    handleForgotPassword,
    setShowForgotPassword,
    authError,
    authSuccess,
    isLoading
}) {
    return (
        <div className="form-box">
            <button className="back-btn" onClick={() => { setShowForgotPassword(false); }}>
                ← Back
            </button>
            <h2>Reset Password</h2>
            <p className="form-subtitle">Enter your email to receive a reset link</p>
            <form onSubmit={handleForgotPassword}>
                <div className="form-group">
                    <label>Email</label>
                    <input 
                        type="email" 
                        value={forgotEmail}
                        onChange={(e) => setForgotEmail(e.target.value)}
                        placeholder="Enter your email"
                        required
                    />
                </div>
                {authError && <div style={{ color: '#ef4444', fontSize: '13px', marginBottom: '12px' }}>{authError}</div>}
                {authSuccess && <div style={{ color: '#22c55e', fontSize: '13px', marginBottom: '12px' }}>{authSuccess}</div>}
                <button type="submit" className="btn-primary" disabled={isLoading}>
                    {isLoading ? 'Sending...' : 'Send Reset Link'}
                </button>
                <p className="form-footer">
                    Remember your password? <a href="#" onClick={(e) => { e.preventDefault(); setShowForgotPassword(false); }}>Sign In</a>
                </p>
            </form>
        </div>
    );
}