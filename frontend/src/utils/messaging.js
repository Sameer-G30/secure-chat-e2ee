// src/utils/messaging.js
import { ref, push, onChildAdded, query, limitToLast, get, update, remove } from 'firebase/database';
import { db } from './encryption';
import { 
    getKeys, 
    deriveSharedSecret, 
    encryptMessage, 
    decryptMessage,
    cleanupEpochKeys,
    clearEpochKeys
} from './encryption';
import { getPublicKey } from './contacts';

const conversationEpochs = new Map();
const MESSAGES_PER_EPOCH = 50;

const getCurrentEpoch = (conversationId) => {
    if (!conversationEpochs.has(conversationId)) {
        conversationEpochs.set(conversationId, 1);
    }
    return conversationEpochs.get(conversationId);
};

const incrementEpoch = (conversationId) => {
    const current = getCurrentEpoch(conversationId);
    const next = current + 1;
    conversationEpochs.set(conversationId, next);
    return next;
};

export const markAsDelivered = async (conversationId, messageId) => {
    try {
        const messageRef = ref(db, `messages/${conversationId}/${messageId}/status`);
        await update(messageRef, {
            delivered: true,
            deliveredAt: Date.now()
        });
        return true;
    } catch (error) {
        console.error('Mark delivered error:', error);
        return false;
    }
};

export const markAsRead = async (conversationId, messageId, userUid) => {
    try {
        const readRef = ref(db, `messages/${conversationId}/${messageId}/readBy`);
        const snapshot = await get(readRef);
        const readBy = snapshot.val() || [];
        
        if (!readBy.includes(userUid)) {
            readBy.push(userUid);
            await set(readRef, readBy);
        }
        
        const statusRef = ref(db, `messages/${conversationId}/${messageId}/status`);
        await update(statusRef, {
            read: true,
            readAt: Date.now()
        });
        
        return true;
    } catch (error) {
        console.error('Mark read error:', error);
        return false;
    }
};

export const sendMessage = async (senderUid, receiverUid, messageText) => {
    try {
        const myKeys = getKeys();
        if (!myKeys) {
            throw new Error('Encryption keys not found');
        }
        
        const receiverPublicKey = await getPublicKey(receiverUid);
        if (!receiverPublicKey) {
            throw new Error('Receiver has no public key');
        }
        
        const masterSecret = await deriveSharedSecret(
            myKeys.privateKey,
            receiverPublicKey
        );
        if (!masterSecret) {
            throw new Error('Failed to derive shared secret');
        }
        
        const conversationId = [senderUid, receiverUid].sort().join('_');
        
        let currentEpoch = getCurrentEpoch(conversationId);
        
        const messagesRef = ref(db, `messages/${conversationId}`);
        const snapshot = await get(messagesRef);
        const messages = snapshot.val();
        let messageCount = messages ? Object.keys(messages).length : 0;
        
        if (messageCount > 0 && messageCount % MESSAGES_PER_EPOCH === 0) {
            currentEpoch = incrementEpoch(conversationId);
        }
        
        const encrypted = await encryptMessage(
            messageText,
            masterSecret,
            conversationId,
            currentEpoch
        );
        if (!encrypted) {
            throw new Error('Encryption failed');
        }
        
        const newMessageRef = await push(messagesRef, {
            sender: senderUid,
            receiver: receiverUid,
            ciphertext: encrypted.ciphertext,
            nonce: encrypted.nonce,
            epoch: encrypted.epoch,
            timestamp: Date.now(),
            status: {
                sent: true,
                delivered: false,
                read: false
            },
            readBy: [],
            deleted: false,
            deletedForEveryone: false
        });
        
        cleanupEpochKeys(conversationId, currentEpoch);
        
        return {
            success: true,
            messageId: newMessageRef.key
        };
    } catch (error) {
        console.error('Send message error:', error);
        return { success: false, error: error.message };
    }
};

export const deleteMessage = async (userUid, conversationId, messageId, deleteForEveryone = false) => {
    try {
        const messageRef = ref(db, `messages/${conversationId}/${messageId}`);
        
        if (deleteForEveryone) {
            await update(messageRef, {
                deletedForEveryone: true,
                deletedAt: Date.now(),
                deletedBy: userUid,
                text: null,
                ciphertext: null,
                nonce: null
            });
        } else {
            await update(messageRef, {
                deleted: true,
                deletedBy: userUid,
                deletedAt: Date.now()
            });
        }
        
        return true;
    } catch (error) {
        console.error('Delete message error:', error);
        return false;
    }
};

export const editMessage = async (userUid, conversationId, messageId, newText) => {
    try {
        const myKeys = getKeys();
        if (!myKeys) {
            throw new Error('Encryption keys not found');
        }
        
        const [uid1, uid2] = conversationId.split('_');
        const otherUid = uid1 === userUid ? uid2 : uid1;
        
        const otherPublicKey = await getPublicKey(otherUid);
        if (!otherPublicKey) {
            throw new Error('Other user has no public key');
        }
        
        const masterSecret = await deriveSharedSecret(
            myKeys.privateKey,
            otherPublicKey
        );
        
        if (!masterSecret) {
            throw new Error('Failed to derive master secret');
        }
        
        const currentEpoch = getCurrentEpoch(conversationId);
        
        const encrypted = await encryptMessage(
            newText,
            masterSecret,
            conversationId,
            currentEpoch
        );
        if (!encrypted) {
            throw new Error('Encryption failed');
        }
        
        const messageRef = ref(db, `messages/${conversationId}/${messageId}`);
        await update(messageRef, {
            ciphertext: encrypted.ciphertext,
            nonce: encrypted.nonce,
            edited: true,
            editedAt: Date.now(),
            epoch: encrypted.epoch
        });
        
        return true;
    } catch (error) {
        console.error('Edit message error:', error);
        return false;
    }
};

export const clearChat = async (conversationId) => {
    try {
        const messagesRef = ref(db, `messages/${conversationId}`);
        await remove(messagesRef);
        conversationEpochs.set(conversationId, 1);
        clearEpochKeys(conversationId);
        return true;
    } catch (error) {
        console.error('Clear chat error:', error);
        return false;
    }
};

export const listenForMessages = (userUid, contactUid, callback) => {
    const conversationId = [userUid, contactUid].sort().join('_');
    const messagesRef = ref(db, `messages/${conversationId}`);
    const messagesQuery = query(messagesRef, limitToLast(50));
    
    const unsubscribe = onChildAdded(messagesQuery, async (snapshot) => {
        const messageData = snapshot.val();
        const messageId = snapshot.key;
        
        if (!messageData) return;
        
        if (messageData.deletedForEveryone) {
            callback({
                id: messageId,
                ...messageData,
                isDeleted: true,
                deletedForEveryone: true,
                text: 'This message was deleted',
                decryptedText: 'This message was deleted'
            }, messageId, messageData.sender === userUid);
            return;
        }
        
        if (messageData.deleted && messageData.deletedBy === userUid) {
            callback({
                id: messageId,
                ...messageData,
                isDeleted: true,
                text: 'You deleted this message',
                decryptedText: 'You deleted this message'
            }, messageId, true);
            return;
        }
        
        const isSent = messageData.sender === userUid;
        
        if (!isSent && messageData.status && !messageData.status.delivered) {
            await markAsDelivered(conversationId, messageId);
        }
        
        try {
            const myKeys = getKeys();
            if (!myKeys) {
                console.error('No keys found');
                return;
            }
            
            const otherUid = messageData.sender === userUid ? contactUid : messageData.sender;
            const otherPublicKey = await getPublicKey(otherUid);
            
            if (!otherPublicKey) {
                console.error('Other user has no public key');
                return;
            }
            
            const masterSecret = await deriveSharedSecret(
                myKeys.privateKey,
                otherPublicKey
            );
            
            if (!masterSecret) {
                console.error('Failed to derive master secret');
                return;
            }
            
            const decrypted = await decryptMessage(
                messageData.ciphertext,
                messageData.nonce,
                masterSecret,
                conversationId,
                messageData.epoch || 1
            );
            
            if (decrypted) {
                callback({
                    id: messageId,
                    ...messageData,
                    decryptedText: decrypted,
                    isSent: isSent,
                    text: decrypted,
                    isDeleted: false,
                    deletedForEveryone: false
                }, messageId, isSent);
            }
        } catch (error) {
            console.error('Decryption error:', error);
        }
    });
    
    return unsubscribe;
};

export const loadMessages = async (userUid, contactUid) => {
    try {
        const conversationId = [userUid, contactUid].sort().join('_');
        const messagesRef = ref(db, `messages/${conversationId}`);
        const snapshot = await get(messagesRef);
        const data = snapshot.val();
        
        if (!data) return [];
        
        const myKeys = getKeys();
        if (!myKeys) {
            console.error('No keys found');
            return [];
        }
        
        const otherPublicKey = await getPublicKey(contactUid);
        if (!otherPublicKey) {
            console.error('Other user has no public key');
            return [];
        }
        
        const masterSecret = await deriveSharedSecret(
            myKeys.privateKey,
            otherPublicKey
        );
        
        const messages = [];
        for (const [id, msg] of Object.entries(data)) {
            if (!msg) continue;
            
            if (msg.deletedForEveryone) {
                messages.push({
                    id: id,
                    ...msg,
                    isSent: msg.sender === userUid,
                    text: 'This message was deleted',
                    isDeleted: true,
                    deletedForEveryone: true
                });
                continue;
            }
            
            if (msg.deleted && msg.deletedBy === userUid) {
                messages.push({
                    id: id,
                    ...msg,
                    isSent: true,
                    text: 'You deleted this message',
                    isDeleted: true
                });
                continue;
            }
            
            try {
                const decrypted = await decryptMessage(
                    msg.ciphertext,
                    msg.nonce,
                    masterSecret,
                    conversationId,
                    msg.epoch || 1
                );
                
                messages.push({
                    id: id,
                    ...msg,
                    isSent: msg.sender === userUid,
                    text: decrypted || 'Decryption failed',
                    isDeleted: false,
                    deletedForEveryone: false
                });
            } catch (error) {
                console.error('Error loading message:', error);
                messages.push({
                    id: id,
                    ...msg,
                    isSent: msg.sender === userUid,
                    text: 'Error loading',
                    isDeleted: false,
                    deletedForEveryone: false
                });
            }
        }
        
        return messages.sort((a, b) => a.timestamp - b.timestamp);
    } catch (error) {
        console.error('Load messages error:', error);
        return [];
    }
};

export const markAllMessagesAsRead = async (userUid, contactUid) => {
    try {
        const conversationId = [userUid, contactUid].sort().join('_');
        const messagesRef = ref(db, `messages/${conversationId}`);
        const snapshot = await get(messagesRef);
        const data = snapshot.val();
        
        if (!data) return true;
        
        for (const [id, msg] of Object.entries(data)) {
            if (msg.sender !== userUid && msg.status && !msg.status.read && !msg.deleted && !msg.deletedForEveryone) {
                await markAsRead(conversationId, id, userUid);
            }
        }
        
        return true;
    } catch (error) {
        console.error('Mark all read error:', error);
        return false;
    }
};