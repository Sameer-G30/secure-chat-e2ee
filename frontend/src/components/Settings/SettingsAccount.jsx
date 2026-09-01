// src/components/Settings/SettingsAccount.jsx
import { useState } from 'react';
import { getAuth, updatePassword, EmailAuthProvider, reauthenticateWithCredential } from 'firebase/auth';
import Modal from '../UI/Modal';

export default function SettingsAccount({ goBackToMain, closeSettings, currentUser }) {
    const auth = getAuth();
    const [showChangePassword, setShowChangePassword] = useState(false);
    const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
    const [currentPassword, setCurrentPassword] = useState('');
    const [newPassword, setNewPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const handleChangePassword = async () => {
        setError('');
        setSuccess('');
        if (newPassword !== confirmPassword) {
            setError('Passwords do not match');
            return;
        }
        if (newPassword.length < 6) {
            setError('Password must be at least 6 characters');
            return;
        }
        setIsLoading(true);
        try {
            const user = auth.currentUser;
            const credential = EmailAuthProvider.credential(user.email, currentPassword);
            await reauthenticateWithCredential(user, credential);
            await updatePassword(user, newPassword);
            setSuccess('Password updated successfully!');
            setCurrentPassword('');
            setNewPassword('');
            setConfirmPassword('');
            setTimeout(() => setShowChangePassword(false), 2000);
        } catch (error) {
            if (error.code === 'auth/wrong-password') {
                setError('Current password is incorrect');
            } else {
                setError(error.message);
            }
        }
        setIsLoading(false);
    };

    const handleDeleteAccount = async () => {
        try {
            const user = auth.currentUser;
            await user.delete();
            alert('Account deleted successfully');
            window.location.reload();
        } catch (error) {
            alert('Error deleting account: ' + error.message);
        }
    };

    return (
        <>
            <div className="settings-header">
                <button className="icon-btn" onClick={goBackToMain}>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polyline points="15 18 9 12 15 6"/>
                    </svg>
                </button>
                <h2>Account</h2>
                <button className="icon-btn" onClick={closeSettings}>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <line x1="18" y1="6" x2="6" y2="18"/>
                        <line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>
            </div>
            <div className="settings-content">
                <div className="settings-item" onClick={() => setShowChangePassword(true)}>
                    <div className="settings-item-left">
                        <div>
                            <div className="settings-item-title">Change Password</div>
                            <div className="settings-item-desc">Update your account password</div>
                        </div>
                    </div>
                    <span className="settings-arrow">›</span>
                </div>

                <div className="settings-item" onClick={() => setShowDeleteConfirm(true)} style={{ color: '#ef4444' }}>
                    <div className="settings-item-left">
                        <div>
                            <div className="settings-item-title">Delete Account</div>
                            <div className="settings-item-desc">Permanently delete your account</div>
                        </div>
                    </div>
                    <span className="settings-arrow">›</span>
                </div>
            </div>

            {showChangePassword && (
                <Modal
                    isOpen={showChangePassword}
                    onClose={() => setShowChangePassword(false)}
                    onConfirm={handleChangePassword}
                    title="Change Password"
                    confirmText="Update"
                    cancelText="Cancel"
                >
                    <div className="modal-body">
                        {error && <div className="error-message">{error}</div>}
                        {success && <div className="success-message">{success}</div>}
                        <div className="form-group">
                            <label>Current Password</label>
                            <input type="password" value={currentPassword} onChange={(e) => setCurrentPassword(e.target.value)} placeholder="Enter current password" />
                        </div>
                        <div className="form-group">
                            <label>New Password</label>
                            <input type="password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} placeholder="Enter new password" />
                        </div>
                        <div className="form-group">
                            <label>Confirm New Password</label>
                            <input type="password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} placeholder="Confirm new password" />
                        </div>
                        <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '8px' }}>Password must be at least 6 characters</p>
                    </div>
                </Modal>
            )}

            {showDeleteConfirm && (
                <Modal
                    isOpen={showDeleteConfirm}
                    onClose={() => setShowDeleteConfirm(false)}
                    onConfirm={handleDeleteAccount}
                    title="Delete Account"
                    message="Are you sure you want to permanently delete your account? This action cannot be undone and all your data will be lost."
                    confirmText="Delete Account"
                    cancelText="Cancel"
                />
            )}
        </>
    );
}