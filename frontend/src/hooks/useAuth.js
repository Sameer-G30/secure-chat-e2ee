// src/hooks/useAuth.js
import { useState, useEffect } from 'react';
import { 
    getAuth, 
    createUserWithEmailAndPassword, 
    signInWithEmailAndPassword,
    sendEmailVerification,
    sendPasswordResetEmail,
    onAuthStateChanged,
    updateProfile,
    signOut
} from 'firebase/auth';
import { ref, set, get } from 'firebase/database';
import { db, generateKeypair, storeKeys, getKeys, hasKeys } from '../utils/encryption';
import { app } from '../utils/encryption';

const auth = getAuth(app);

export function useAuth() {
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const [currentUser, setCurrentUser] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [authError, setAuthError] = useState('');
    const [authSuccess, setAuthSuccess] = useState('');
    const [showVerificationSent, setShowVerificationSent] = useState(false);
    const [pendingUserEmail, setPendingUserEmail] = useState('');

    // Check auth state
    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, async (user) => {
            if (user) {
                if (user.emailVerified) {
                    setCurrentUser(user);
                    setIsLoggedIn(true);
                    setShowVerificationSent(false);
                } else {
                    setPendingUserEmail(user.email);
                    setCurrentUser(user);
                    setIsLoggedIn(false);
                    setShowVerificationSent(true);
                    
                    if (!localStorage.getItem('verificationEmailSent')) {
                        try {
                            await sendEmailVerification(user);
                            localStorage.setItem('verificationEmailSent', 'true');
                        } catch (error) {
                            console.error('Error sending verification:', error);
                        }
                    }
                }
            } else {
                setCurrentUser(null);
                setIsLoggedIn(false);
                setShowVerificationSent(false);
                localStorage.removeItem('verificationEmailSent');
            }
            setIsLoading(false);
        });

        return () => unsubscribe();
    }, []);

    // Auto-verification check
    useEffect(() => {
        if (!showVerificationSent || !currentUser) return;

        const interval = setInterval(async () => {
            try {
                await currentUser.reload();
                const updatedUser = auth.currentUser;
                if (updatedUser && updatedUser.emailVerified) {
                    setCurrentUser(updatedUser);
                    setIsLoggedIn(true);
                    setShowVerificationSent(false);
                    clearInterval(interval);
                }
            } catch (error) {
                console.error('Error checking verification:', error);
            }
        }, 3000);

        return () => clearInterval(interval);
    }, [showVerificationSent, currentUser]);

    // Ensure keys exist
    const ensureKeysExist = async () => {
        if (hasKeys()) {
            console.log('✅ Keys already exist');
            return true;
        }
        
        try {
            console.log('🔄 Generating new keys...');
            const keypair = await generateKeypair();
            storeKeys(keypair.publicKey, keypair.privateKey);
            console.log('✅ Keys generated and stored');
            return true;
        } catch (error) {
            console.error('❌ Failed to generate keys:', error);
            return false;
        }
    };

    // Register
    const handleRegister = async (registerName, registerEmail, registerPassword, passwordStrength) => {
        setAuthError('');
        setAuthSuccess('');
        
        if (!passwordStrength.isStrong) {
            setAuthError('Please use a stronger password');
            return;
        }
        
        setIsLoading(true);
        
        try {
            // Check if username is taken
            const usersRef = ref(db, 'users');
            const snapshot = await get(usersRef);
            const users = snapshot.val();
            
            if (users) {
                for (const [uid, userData] of Object.entries(users)) {
                    if (userData.username && userData.username.toLowerCase() === registerName.toLowerCase()) {
                        setAuthError('Username already taken. Please choose another.');
                        setIsLoading(false);
                        return;
                    }
                }
            }
            
            const userCredential = await createUserWithEmailAndPassword(
                auth, 
                registerEmail, 
                registerPassword
            );
            
            const user = userCredential.user;
            
            await updateProfile(user, {
                displayName: registerName
            });
            
            const keypair = await generateKeypair();
            storeKeys(keypair.publicKey, keypair.privateKey);
            
            const userData = {
                username: registerName,
                email: registerEmail,
                publicKey: keypair.publicKey,
                createdAt: Date.now()
            };
            
            await set(ref(db, 'users/' + user.uid), userData);
            
            await sendEmailVerification(user);
            
            setPendingUserEmail(registerEmail);
            setShowVerificationSent(true);
            localStorage.setItem('verificationEmailSent', 'true');
            setAuthSuccess('Account created! Please verify your email.');
            
        } catch (error) {
            console.error('❌ Registration error:', error);
            if (error.code === 'auth/email-already-in-use') {
                setAuthError('This email is already registered. Please sign in instead.');
            } else if (error.code === 'auth/invalid-email') {
                setAuthError('Please enter a valid email address.');
            } else if (error.code === 'auth/too-many-requests') {
                setAuthError('Too many attempts. Please wait 10 minutes and try again.');
            } else {
                setAuthError('Registration failed: ' + error.message);
            }
        }
        
        setIsLoading(false);
    };

    // Login
    const handleLogin = async (loginEmail, loginPassword) => {
        setAuthError('');
        setAuthSuccess('');
        
        if (!loginEmail || !loginPassword) {
            setAuthError('Please enter both email and password');
            return;
        }
        
        setIsLoading(true);
        
        try {
            const userCredential = await signInWithEmailAndPassword(
                auth, 
                loginEmail, 
                loginPassword
            );
            
            const user = userCredential.user;
            
            if (!user.emailVerified) {
                await sendEmailVerification(user);
                setPendingUserEmail(user.email);
                setShowVerificationSent(true);
                setAuthError('Please verify your email first.');
                await signOut(auth);
                setIsLoading(false);
                return;
            }
            
            const keysGenerated = await ensureKeysExist();
            if (!keysGenerated) {
                setAuthError('Failed to generate encryption keys.');
                setIsLoading(false);
                return;
            }
            
            const keys = getKeys();
            if (keys) {
                await set(ref(db, 'users/' + user.uid + '/publicKey'), keys.publicKey);
            }
            
        } catch (error) {
            if (error.code === 'auth/invalid-credential' || error.code === 'auth/user-not-found') {
                setAuthError('Invalid email or password.');
            } else if (error.code === 'auth/too-many-requests') {
                setAuthError('Too many failed attempts. Please try again later.');
            } else {
                setAuthError(error.message);
            }
        }
        
        setIsLoading(false);
    };

    // Resend verification
    const resendVerification = async () => {
        if (!currentUser && !pendingUserEmail) {
            setAuthError('No user found.');
            return;
        }
        
        setAuthError('');
        setAuthSuccess('');
        
        try {
            if (currentUser) {
                await sendEmailVerification(currentUser);
                setAuthSuccess('Verification email resent.');
                setTimeout(() => setAuthSuccess(''), 3000);
            } else {
                setAuthError('Please try registering again.');
            }
        } catch (error) {
            setAuthError('Failed to send verification email.');
        }
    };

    // Forgot password
    const handleForgotPassword = async (forgotEmail) => {
        setAuthError('');
        setAuthSuccess('');
        
        if (!forgotEmail) {
            setAuthError('Please enter your email');
            return;
        }
        
        setIsLoading(true);
        
        try {
            await sendPasswordResetEmail(auth, forgotEmail, {
                url: window.location.origin,
                handleCodeInApp: false,
            });
            
            setAuthSuccess('Password reset email sent.');
            
            setTimeout(() => {
                setAuthSuccess('');
            }, 3000);
            
        } catch (error) {
            if (error.code === 'auth/user-not-found') {
                setAuthError('No account found with this email.');
            } else {
                setAuthError(error.message);
            }
        }
        
        setIsLoading(false);
    };

    // Logout
    const handleLogout = async () => {
        try {
            await signOut(auth);
            setCurrentUser(null);
            setIsLoggedIn(false);
            setShowVerificationSent(false);
        } catch (error) {
            console.error('Logout error:', error);
        }
    };

    return {
        auth,
        isLoggedIn,
        currentUser,
        isLoading,
        authError,
        authSuccess,
        showVerificationSent,
        pendingUserEmail,
        setShowRegister: (val) => {}, // Will be handled in component
        setShowForgotPassword: (val) => {},
        handleRegister,
        handleLogin,
        handleForgotPassword,
        resendVerification,
        handleLogout,
        setAuthError,
        setAuthSuccess
    };
}