// src/utils/reports.js
import { ref, push, set } from 'firebase/database';
import { db } from './encryption';

export const submitReport = async (reporterUid, reportedUid, reason, blockOnReport = false) => {
    try {
        // Check if we have a valid database connection
        if (!db) {
            console.error('❌ Database not initialized');
            return false;
        }

        const reportRef = ref(db, 'reports');
        const newReportRef = push(reportRef);
        
        await set(newReportRef, {
            reporterUid: reporterUid,
            reportedUid: reportedUid,
            reason: reason,
            blockOnReport: blockOnReport,
            timestamp: Date.now(),
            status: 'pending'
        });
        
        console.log('✅ Report submitted successfully');
        return true;
    } catch (error) {
        console.error('❌ Error submitting report:', error);
        return false;
    }
};