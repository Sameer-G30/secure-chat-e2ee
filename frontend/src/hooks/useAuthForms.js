// src/hooks/useAuthForms.js
import { useState } from 'react';

export function useAuthForms() {
    const [showRegister, setShowRegister] = useState(false);
    const [showForgotPassword, setShowForgotPassword] = useState(false);
    const [loginEmail, setLoginEmail] = useState('');
    const [loginPassword, setLoginPassword] = useState('');
    const [registerName, setRegisterName] = useState('');
    const [registerEmail, setRegisterEmail] = useState('');
    const [registerPassword, setRegisterPassword] = useState('');
    const [registerConfirmPassword, setRegisterConfirmPassword] = useState('');
    const [forgotEmail, setForgotEmail] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [passwordStrength, setPasswordStrength] = useState({
        score: 0,
        label: '',
        color: '',
        requirements: {
            length: false,
            uppercase: false,
            lowercase: false,
            number: false,
            special: false,
        }
    });

    const checkPasswordStrength = (password) => {
        const requirements = {
            length: password.length >= 8,
            uppercase: /[A-Z]/.test(password),
            lowercase: /[a-z]/.test(password),
            number: /[0-9]/.test(password),
            special: /[^A-Za-z0-9]/.test(password),
        };
        
        const score = Object.values(requirements).filter(Boolean).length;
        const labels = ['Very Weak', 'Weak', 'Fair', 'Good', 'Strong'];
        const colors = ['#ef4444', '#f59e0b', '#f59e0b', '#22c55e', '#22c55e'];
        
        setPasswordStrength({
            score,
            label: labels[score],
            color: colors[score],
            requirements,
            isStrong: score >= 4,
        });
    };

    return {
        showRegister,
        setShowRegister,
        showForgotPassword,
        setShowForgotPassword,
        loginEmail,
        setLoginEmail,
        loginPassword,
        setLoginPassword,
        registerName,
        setRegisterName,
        registerEmail,
        setRegisterEmail,
        registerPassword,
        setRegisterPassword,
        registerConfirmPassword,
        setRegisterConfirmPassword,
        forgotEmail,
        setForgotEmail,
        showPassword,
        setShowPassword,
        passwordStrength,
        checkPasswordStrength
    };
}