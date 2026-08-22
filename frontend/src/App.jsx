// src/App.jsx
import { useRef } from 'react';

// ===== COMPONENTS =====
import AuthScreen from './components/Auth/AuthScreen';
import Sidebar from './components/Contacts/Sidebar';
import ChatWindow from './components/Chat/ChatWindow';
import SettingsPanel from './components/Settings/SettingsPanel';

// ===== HOOKS =====
import { useAuth } from './hooks/useAuth';
import { useAuthForms } from './hooks/useAuthForms';
import { useContacts } from './hooks/useContacts';
import { useChat } from './hooks/useChat';
import { useBlock } from './hooks/useBlock';
import { useTheme } from './hooks/useTheme';
import { useSettings } from './hooks/useSettings';

function App() {
    // ===== HOOKS =====
    const auth = useAuth();
    const forms = useAuthForms();
    const block = useBlock();
    const contacts = useContacts(auth.currentUser);
    
    const isBlocked = contacts.selectedContact ? block.isUserBlocked(contacts.selectedContact.uid) : false;
    const chat = useChat(auth.currentUser, contacts.selectedContact, isBlocked);
    
    const theme = useTheme();
    const settings = useSettings();

    const messagesEndRef = useRef(null);

    // ============ LOADING SCREEN ============
    if (auth.isLoading) {
        return (
            <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                height: '100vh',
                background: '#0f0f13',
                color: 'white',
            }}>
                <div style={{ textAlign: 'center' }}>
                    <div style={{
                        width: '40px',
                        height: '40px',
                        border: '3px solid #2d2d3d',
                        borderTopColor: '#4f46e5',
                        borderRadius: '50%',
                        animation: 'spin 0.8s linear infinite',
                        margin: '0 auto 16px',
                    }} />
                    <p style={{ color: '#64748b', fontSize: '14px' }}>Loading...</p>
                </div>
                <style>{`
                    @keyframes spin {
                        to { transform: rotate(360deg); }
                    }
                `}</style>
            </div>
        );
    }

    // ============ AUTH SCREENS ============
    if (auth.showVerificationSent || !auth.isLoggedIn) {
        return (
            <div className="auth-screen">
                <div className="auth-container">
                    <div className="auth-brand">
                        <div className="brand-icon">
                            <svg width="52" height="52" viewBox="0 0 52 52" fill="none">
                                <rect width="52" height="52" rx="14" fill="#4f46e5"/>
                                <path d="M26 15L26 37M15 26L37 26" stroke="white" strokeWidth="3.5" strokeLinecap="round"/>
                                <circle cx="26" cy="26" r="9" stroke="white" strokeWidth="3.5"/>
                            </svg>
                        </div>
                        <h1>Secure<span>Chat</span></h1>
                        <p>End-to-end encrypted messaging.<br/>Privacy by design.</p>
                        <div className="trust-badge">
                            <span>End-to-end encrypted · No data stored</span>
                        </div>
                    </div>

                    <div className="auth-form">
                        <AuthScreen
                            showRegister={forms.showRegister}
                            setShowRegister={forms.setShowRegister}
                            showForgotPassword={forms.showForgotPassword}
                            setShowForgotPassword={forms.setShowForgotPassword}
                            showVerificationSent={auth.showVerificationSent}
                            pendingUserEmail={auth.pendingUserEmail}
                            currentUser={auth.currentUser}
                            loginEmail={forms.loginEmail}
                            setLoginEmail={forms.setLoginEmail}
                            loginPassword={forms.loginPassword}
                            setLoginPassword={forms.setLoginPassword}
                            handleLogin={() => auth.handleLogin(forms.loginEmail, forms.loginPassword)}
                            registerName={forms.registerName}
                            setRegisterName={forms.setRegisterName}
                            registerEmail={forms.registerEmail}
                            setRegisterEmail={forms.setRegisterEmail}
                            registerPassword={forms.registerPassword}
                            setRegisterPassword={forms.setRegisterPassword}
                            registerConfirmPassword={forms.registerConfirmPassword}
                            setRegisterConfirmPassword={forms.setRegisterConfirmPassword}
                            handleRegister={() => auth.handleRegister(
                                forms.registerName,
                                forms.registerEmail,
                                forms.registerPassword,
                                forms.passwordStrength
                            )}
                            passwordStrength={forms.passwordStrength}
                            checkPasswordStrength={forms.checkPasswordStrength}
                            showPassword={forms.showPassword}
                            setShowPassword={forms.setShowPassword}
                            forgotEmail={forms.forgotEmail}
                            setForgotEmail={forms.setForgotEmail}
                            handleForgotPassword={() => auth.handleForgotPassword(forms.forgotEmail)}
                            authError={auth.authError}
                            authSuccess={auth.authSuccess}
                            isLoading={auth.isLoading}
                            resendVerification={auth.resendVerification}
                        />
                    </div>
                </div>
            </div>
        );
    }

    // ============ CHAT APP ============
    return (
        <div className="chat-app">
            <Sidebar
                currentUser={auth.currentUser}
                openSettings={settings.openSettings}
                searchTerm={contacts.searchTerm}
                setSearchTerm={contacts.setSearchTerm}
                handleSearchUsers={contacts.handleSearchUsers}
                isSearching={contacts.isSearching}
                searchResults={contacts.searchResults}
                handleAddContact={contacts.handleAddContact}
                contactList={contacts.contactList}
                selectedContact={contacts.selectedContact}
                setSelectedContact={contacts.setSelectedContact}
            />

            <div className="chat-area">
                <ChatWindow
                    selectedContact={contacts.selectedContact}
                    chatMessages={chat.chatMessages}
                    isLoadingMessages={chat.isLoadingMessages}
                    messageInput={chat.messageInput}
                    setMessageInput={chat.setMessageInput}
                    handleSendMessage={chat.handleSendMessage}
                    showMessageActions={chat.showMessageActions}
                    selectedMessage={chat.selectedMessage}
                    closeMessageActions={chat.closeMessageActions}
                    handleDeleteMessage={chat.handleDeleteMessage}
                    startEditing={chat.startEditing}
                    isEditing={chat.isEditing}
                    editText={chat.editText}
                    setEditText={chat.setEditText}
                    saveEdit={chat.saveEdit}
                    showClearChatConfirmation={chat.showClearChatConfirmation}
                    setShowClearChatConfirmation={chat.setShowClearChatConfirmation}
                    handleClearChat={chat.handleClearChat}
                    messagesEndRef={messagesEndRef}
                    openMessageActions={chat.openMessageActions}
                    isBlocked={isBlocked}
                    onBlockUser={block.blockUser}
                    onUnblockUser={block.unblockUser}
                    currentUser={auth.currentUser}
                />
            </div>

            {settings.showSettings && (
                <SettingsPanel
                    settingsView={settings.settingsView}
                    setSettingsView={settings.setSettingsView}
                    closeSettings={settings.closeSettings}
                    handleLogout={auth.handleLogout}
                    goToGeneral={settings.goToGeneral}
                    goBackToMain={settings.goBackToMain}
                    goBackToGeneral={settings.goBackToGeneral}
                    goToTheme={settings.goToTheme}
                    selectedTheme={theme.selectedTheme}
                    applyTheme={theme.applyTheme}
                />
            )}
        </div>
    );
}

export default App;