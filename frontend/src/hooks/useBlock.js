// src/hooks/useBlock.js
import { useState, useEffect } from 'react';

export function useBlock() {
    const [blockedUsers, setBlockedUsers] = useState(() => {
        const saved = localStorage.getItem('blockedUsers');
        return saved ? JSON.parse(saved) : [];
    });

    useEffect(() => {
        localStorage.setItem('blockedUsers', JSON.stringify(blockedUsers));
    }, [blockedUsers]);

    const isUserBlocked = (userId) => {
        return blockedUsers.includes(userId);
    };

    const blockUser = (userId) => {
        if (!blockedUsers.includes(userId)) {
            setBlockedUsers([...blockedUsers, userId]);
        }
    };

    const unblockUser = (userId) => {
        setBlockedUsers(blockedUsers.filter(id => id !== userId));
    };

    return {
        blockedUsers,
        isUserBlocked,
        blockUser,
        unblockUser
    };
}