// src/components/Chat/ContactProfile.jsx
import { useState, useEffect } from 'react';
import { ref, get } from 'firebase/database';
import { db } from '../../utils/encryption';

export default function ContactProfile({ 
    isOpen, 
    onClose, 
    contact, 
    currentUser,
    profilePicture 
}) {
    const [contactProfilePic, setContactProfilePic] = useState(null);
    const [contactBio, setContactBio] = useState('');
    const [contactLastSeen, setContactLastSeen] = useState(null);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        if (!isOpen || !contact) return;

        const fetchContactData = async () => {
            setIsLoading(true);
            try {
                const userRef = ref(db, 'users/' + contact.uid);
                const snapshot = await get(userRef);
                const data = snapshot.val();
                
                if (data) {
                    const savedPic = localStorage.getItem('profilePicture_' + contact.uid);
                    if (savedPic) {
                        setContactProfilePic(savedPic);
                    } else if (data.profilePicture) {
                        setContactProfilePic(data.profilePicture);
                    }
                    
                    setContactBio(data.bio || 'Hey there! I am using SecureChat');
                    setContactLastSeen(data.lastSeen || null);
                }
            } catch (error) {
                console.error('Error fetching contact data:', error);
            }
            setIsLoading(false);
        };

        fetchContactData();
    }, [isOpen, contact]);

    if (!isOpen || !contact) return null;

    const formatLastSeen = (timestamp) => {
        if (!timestamp) return 'Recently';
        const now = Date.now();
        const diff = now - timestamp;
        const minutes = Math.floor(diff / 60000);
        const hours = Math.floor(diff / 3600000);
        const days = Math.floor(diff / 86400000);

        if (minutes < 1) return 'Online now';
        if (minutes < 60) return `Last seen ${minutes} min ago`;
        if (hours < 24) return `Last seen ${hours} hr ago`;
        return `Last seen ${days} days ago`;
    };

    return (
        <div className="profile-modal-overlay" onClick={onClose}>
            <div className="profile-modal" onClick={(e) => e.stopPropagation()}>
                <button className="profile-modal-close" onClick={onClose}>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <line x1="18" y1="6" x2="6" y2="18"/>
                        <line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>

                {isLoading ? (
                    <div className="profile-loading">Loading...</div>
                ) : (
                    <>
                        <div className="profile-avatar">
                            {contactProfilePic ? (
                                <img src={contactProfilePic} alt={contact.username} />
                            ) : (
                                <span className="profile-avatar-text">
                                    {contact.username?.charAt(0).toUpperCase() || '?'}
                                </span>
                            )}
                        </div>

                        <div className="profile-name">{contact.username}</div>
                        <div className="profile-username">@{contact.username}</div>

                        <div className="profile-divider"></div>

                        <div className="profile-section">
                            <div className="profile-section-label">About</div>
                            <div className="profile-section-value">{contactBio}</div>
                        </div>

                        <div className="profile-section">
                            <div className="profile-section-label">Last Seen</div>
                            <div className="profile-section-value">
                                {formatLastSeen(contactLastSeen)}
                            </div>
                        </div>

                        <div className="profile-actions">
                            <button className="profile-action-btn" onClick={() => alert('Message sent!')}>
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
                                </svg>
                                Message
                            </button>
                            <button className="profile-action-btn" onClick={() => alert('Voice call coming soon!')}>
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M22 16.92v3a2 2 0 01-2.18 2 19.79 19.79 0 01-8.63-3.07 19.5 19.5 0 01-6-6 19.79 19.79 0 01-3.07-8.67A2 2 0 014.11 2h3a2 2 0 012 1.72c.127.96.361 1.903.7 2.81a2 2 0 01-.45 2.11L8.09 9.91a16 16 0 006 6l1.27-1.27a2 2 0 012.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0122 16.92z"/>
                                </svg>
                                Call
                            </button>
                            <button className="profile-action-btn" onClick={() => alert('Video call coming soon!')}>
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M23 7l-7 5 7 5V7z"/>
                                    <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
                                </svg>
                                Video
                            </button>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}