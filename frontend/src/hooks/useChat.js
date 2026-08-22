// src/hooks/useChat.js
import { useState, useEffect, useRef } from 'react';
import { 
    sendMessage, 
    listenForMessages, 
    loadMessages,
    deleteMessage,
    editMessage,
    clearChat
} from '../utils/messaging';

export function useChat(currentUser, selectedContact, isBlocked = false) {
    const [chatMessages, setChatMessages] = useState([]);
    const [messageInput, setMessageInput] = useState('');
    const [isLoadingMessages, setIsLoadingMessages] = useState(false);
    const [selectedMessage, setSelectedMessage] = useState(null);
    const [showMessageActions, setShowMessageActions] = useState(false);
    const [showClearChatConfirmation, setShowClearChatConfirmation] = useState(false);
    const [isEditing, setIsEditing] = useState(false);
    const [editText, setEditText] = useState('');
    
    // Track if messages have been loaded
    const messagesLoadedRef = useRef(false);

    // Load and listen for messages
    useEffect(() => {
        if (!selectedContact || !currentUser) return;
        
        setChatMessages([]);
        messagesLoadedRef.current = false;
        let isMounted = true;
        let unsubscribe = null;
        
        const loadPrevMessages = async () => {
            setIsLoadingMessages(true);
            try {
                const msgs = await loadMessages(currentUser.uid, selectedContact.uid);
                // ✅ FIX: Show ALL past messages regardless of block status
                if (isMounted) {
                    setChatMessages(msgs);
                    messagesLoadedRef.current = true;
                }
            } catch (error) {
                console.error('Load messages error:', error);
            }
            setIsLoadingMessages(false);
        };
        
        loadPrevMessages();
        
        const setupListener = () => {
            unsubscribe = listenForMessages(
                currentUser.uid,
                selectedContact.uid,
                (messageData, messageId, isSent) => {
                    if (!isMounted) return;
                    
                    // ✅ FIX: Only filter NEW messages from blocked user
                    // Past messages remain visible (they're already in chatMessages)
                    if (isBlocked && !isSent) {
                        // New message from blocked user → ignore it
                        return;
                    }
                    
                    setChatMessages(prev => {
                        const exists = prev.some(msg => msg.id === messageId);
                        if (exists) return prev;
                        
                        return [...prev, {
                            id: messageId,
                            sender: messageData.sender,
                            text: messageData.decryptedText || 'Message',
                            timestamp: messageData.timestamp,
                            isSent: isSent,
                            isDeleted: messageData.isDeleted || false,
                            isEdited: messageData.isEdited || false
                        }];
                    });
                }
            );
        };
        
        setupListener();
        
        return () => {
            isMounted = false;
            if (unsubscribe) unsubscribe();
        };
    }, [selectedContact, currentUser, isBlocked]);

    // Send message - block if user is blocked
    const handleSendMessage = async () => {
        if (!messageInput.trim() || !selectedContact || !currentUser) return;
        
        // Don't allow sending if blocked
        if (isBlocked) {
            alert('You cannot send messages to a blocked user.');
            return;
        }
        
        try {
            const success = await sendMessage(
                currentUser.uid,
                selectedContact.uid,
                messageInput.trim()
            );
            
            if (success) {
                setMessageInput('');
            }
        } catch (error) {
            console.error('Send message error:', error);
            alert('Failed to send message');
        }
    };

    // Message actions
    const openMessageActions = (msg) => {
        setSelectedMessage(msg);
        setShowMessageActions(true);
    };

    const closeMessageActions = () => {
        setShowMessageActions(false);
        setSelectedMessage(null);
    };

    const handleDeleteMessage = async (deleteForEveryone = false) => {
        if (!selectedMessage || !currentUser || !selectedContact) return;
        
        const conversationId = [currentUser.uid, selectedContact.uid].sort().join('_');
        
        const success = await deleteMessage(
            currentUser.uid,
            conversationId,
            selectedMessage.id,
            deleteForEveryone
        );
        
        if (success) {
            setShowMessageActions(false);
            setSelectedMessage(null);
        } else {
            alert('Failed to delete message');
        }
    };

    const startEditing = () => {
        if (!selectedMessage) return;
        setEditText(selectedMessage.text);
        setIsEditing(true);
        setShowMessageActions(false);
    };

    const saveEdit = async () => {
        if (!editText.trim() || !selectedMessage || !currentUser || !selectedContact) return;
        
        const conversationId = [currentUser.uid, selectedContact.uid].sort().join('_');
        
        const success = await editMessage(
            currentUser.uid,
            conversationId,
            selectedMessage.id,
            editText.trim()
        );
        
        if (success) {
            setIsEditing(false);
            setEditText('');
            setSelectedMessage(null);
        } else {
            alert('Failed to edit message');
        }
    };

    const handleClearChat = async () => {
        if (!currentUser || !selectedContact) return;
        
        const conversationId = [currentUser.uid, selectedContact.uid].sort().join('_');
        
        const success = await clearChat(conversationId);
        
        if (success) {
            setShowClearChatConfirmation(false);
            setChatMessages([]);
        } else {
            alert('Failed to clear chat');
        }
    };

    return {
        chatMessages,
        messageInput,
        setMessageInput,
        isLoadingMessages,
        selectedMessage,
        showMessageActions,
        showClearChatConfirmation,
        isEditing,
        editText,
        setEditText,
        handleSendMessage,
        openMessageActions,
        closeMessageActions,
        handleDeleteMessage,
        startEditing,
        saveEdit,
        handleClearChat,
        setShowClearChatConfirmation
    };
}