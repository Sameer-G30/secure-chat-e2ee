// src/components/Auth/AuthScreen.jsx
import Login from './Login';
import Register from './Register';
import ForgotPassword from './ForgotPassword';
import VerifyEmail from './VerifyEmail';

export default function AuthScreen({
    // Auth state
    showRegister,
    setShowRegister,
    showForgotPassword,
    setShowForgotPassword,
    showVerificationSent,
    pendingUserEmail,
    currentUser,
    
    // Login props
    loginEmail,
    setLoginEmail,
    loginPassword,
    setLoginPassword,
    handleLogin,
    
    // Register props
    registerName,
    setRegisterName,
    registerEmail,
    setRegisterEmail,
    registerPassword,
    setRegisterPassword,
    registerConfirmPassword,
    setRegisterConfirmPassword,
    handleRegister,
    passwordStrength,
    checkPasswordStrength,
    showPassword,
    setShowPassword,
    
    // Forgot password props
    forgotEmail,
    setForgotEmail,
    handleForgotPassword,
    
    // Common
    authError,
    authSuccess,
    isLoading,
    resendVerification
}) {
    // If verification screen
    if (showVerificationSent) {
        return (
            <VerifyEmail
                pendingUserEmail={pendingUserEmail}
                currentUser={currentUser}
                authError={authError}
                authSuccess={authSuccess}
                resendVerification={resendVerification}
            />
        );
    }

    // If forgot password
    if (showForgotPassword) {
        return (
            <ForgotPassword
                forgotEmail={forgotEmail}
                setForgotEmail={setForgotEmail}
                handleForgotPassword={handleForgotPassword}
                setShowForgotPassword={setShowForgotPassword}
                authError={authError}
                authSuccess={authSuccess}
                isLoading={isLoading}
            />
        );
    }

    // If register
    if (showRegister) {
        return (
            <Register
                registerName={registerName}
                setRegisterName={setRegisterName}
                registerEmail={registerEmail}
                setRegisterEmail={setRegisterEmail}
                registerPassword={registerPassword}
                setRegisterPassword={setRegisterPassword}
                registerConfirmPassword={registerConfirmPassword}
                setRegisterConfirmPassword={setRegisterConfirmPassword}
                handleRegister={handleRegister}
                setShowRegister={setShowRegister}
                authError={authError}
                authSuccess={authSuccess}
                isLoading={isLoading}
                passwordStrength={passwordStrength}
                checkPasswordStrength={checkPasswordStrength}
                showPassword={showPassword}
                setShowPassword={setShowPassword}
            />
        );
    }

    // Default: Login
    return (
        <Login
            loginEmail={loginEmail}
            setLoginEmail={setLoginEmail}
            loginPassword={loginPassword}
            setLoginPassword={setLoginPassword}
            handleLogin={handleLogin}
            setShowForgotPassword={setShowForgotPassword}
            setShowRegister={setShowRegister}
            authError={authError}
            authSuccess={authSuccess}
            isLoading={isLoading}
        />
    );
}