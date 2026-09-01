// src/components/Settings/SettingsProfile.jsx
import { useState, useEffect } from 'react';
import { ref, update, get } from 'firebase/database';
import { updateProfile } from 'firebase/auth';
import { db } from '../../utils/encryption';

export default function SettingsProfile({ goBackToMain, closeSettings, currentUser, profilePicture, onProfilePictureUpload }) {
    const [username, setUsername] = useState('');
    const [bio, setBio] = useState('');
    const [email, setEmail] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');

    useEffect(() => {
        if (!currentUser) return;
        setEmail(currentUser.email || '');
        setUsername(currentUser.displayName || currentUser.email?.split('@')[0] || '');
        const userRef = ref(db, `users/${currentUser.uid}/bio`);
        get(userRef).then(snapshot => {
            if (snapshot.exists()) {
                setBio(snapshot.val());
            }
        }).catch(console.error);
    }, [currentUser]);

    const handleSaveProfile = async () => {
        if (!username.trim()) {
            setError('Username cannot be empty');
            return;
        }
        setIsLoading(true);
        setError('');
        setSuccess('');
        try {
            await updateProfile(currentUser, { displayName: username });
            await update(ref(db, `users/${currentUser.uid}`), {
                username: username,
                bio: bio,
                updatedAt: Date.now()
            });
            setSuccess('Profile updated successfully!');
            setTimeout(() => setSuccess(''), 3000);
        } catch (error) {
            console.error('Error updating profile:', error);
            setError('Failed to update profile');
        }
        setIsLoading(false);
    };

    return (
        <>
            <div className="settings-header">
                <button className="icon-btn" onClick={goBackToMain}>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polyline points="15 18 9 12 15 6"/>
                    </svg>
                </button>
                <h2>Profile</h2>
                <button className="icon-btn" onClick={closeSettings}>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <line x1="18" y1="6" x2="6" y2="18"/>
                        <line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>
            </div>
            <div className="settings-content">
                {/* Profile Picture - Smaller */}
                <div className="profile-picture-section">
                    <div className="profile-picture-wrapper">
                        {profilePicture ? (
                            <img src={profilePicture} alt="Profile" className="profile-picture-large" />
                        ) : (
                            <div className="profile-picture-placeholder">
                                {username?.charAt(0).toUpperCase() || 'U'}
                            </div>
                        )}
                        <button className="profile-picture-edit" onClick={() => document.getElementById('profilePicInput2').click()}>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                                <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                            </svg>
                        </button>
                    </div>
                    <p className="profile-picture-hint">Tap to change profile photo</p>
                    <input 
                        type="file" 
                        id="profilePicInput2" 
                        accept="image/*" 
                        style={{ display: 'none' }} 
                        onChange={onProfilePictureUpload} 
                    />
                </div>

                <div className="profile-divider"></div>

                <div className="form-group">
                    <label>Display Name</label>
                    <input 
                        type="text" 
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        placeholder="Your name"
                    />
                </div>

                <div className="form-group">
                    <label>Email</label>
                    <input 
                        type="email" 
                        value={email}
                        disabled
                        style={{ opacity: 0.6, cursor: 'not-allowed' }}
                    />
                    <p className="form-hint">Email cannot be changed</p>
                </div>

                <div className="form-group">
                    <label>About / Bio</label>
                    <textarea 
                        value={bio}
                        onChange={(e) => setBio(e.target.value)}
                        placeholder="Tell people about yourself..."
                        rows={3}
                        maxLength={150}
                    />
                    <p className="form-hint">{bio.length}/150 characters</p>
                </div>

                {error && <div className="error-message">{error}</div>}
                {success && <div className="success-message">{success}</div>}

                <button 
                    className="btn-primary" 
                    onClick={handleSaveProfile}
                    disabled={isLoading}
                >
                    {isLoading ? 'Saving...' : 'Save Profile'}
                </button>
            </div>
        </>
    );
}