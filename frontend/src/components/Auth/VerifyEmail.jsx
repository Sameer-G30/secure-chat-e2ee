// src/components/Auth/VerifyEmail.jsx
export default function VerifyEmail({
    pendingUserEmail,
    currentUser,
    authError,
    authSuccess,
    resendVerification
}) {
    return (
        <div className="form-box">
            <div style={{ textAlign: 'center', padding: '20px 0' }}>
                <div style={{ fontSize: '56px', marginBottom: '16px' }}>📧</div>
                <h2>Verify Your Email</h2>
                <p className="form-subtitle">
                    We've sent a verification link to <strong>{pendingUserEmail || currentUser?.email || 'your email'}</strong>
                </p>
                <p style={{ color: 'var(--text-muted)', fontSize: '13px', marginBottom: '20px' }}>
                    Click the link in the email to verify your account. You'll be automatically logged in.
                </p>
                
                {authError && (
                    <div style={{ 
                        color: '#ef4444', 
                        fontSize: '13px', 
                        marginBottom: '12px', 
                        padding: '10px 12px', 
                        background: 'rgba(239,68,68,0.08)',
                        borderRadius: '8px',
                        borderLeft: '3px solid #ef4444',
                        textAlign: 'left'
                    }}>
                        {authError}
                    </div>
                )}
                
                {authSuccess && (
                    <div style={{ 
                        color: '#22c55e', 
                        fontSize: '13px', 
                        marginBottom: '12px', 
                        padding: '10px 12px', 
                        background: 'rgba(34,197,94,0.08)',
                        borderRadius: '8px',
                        borderLeft: '3px solid #22c55e',
                        textAlign: 'left'
                    }}>
                        {authSuccess}
                    </div>
                )}
                
                <div className="verification-spinner">
                    <div className="spinner"></div>
                    <span>Waiting for verification...</span>
                </div>
                
                <p className="form-footer" style={{ marginTop: '12px' }}>
                    Didn't receive the email? <a href="#" onClick={(e) => {
                        e.preventDefault();
                        resendVerification();
                    }}>Resend</a>
                </p>
                
                <p style={{ 
                    marginTop: '16px', 
                    fontSize: '12px', 
                    color: 'var(--text-muted)',
                    borderTop: '1px solid var(--border-color)',
                    paddingTop: '16px'
                }}>
                    We'll automatically detect when you click the verification link.
                </p>
            </div>
        </div>
    );
}