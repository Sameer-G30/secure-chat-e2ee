// src/components/UI/Modal.jsx
export default function Modal({
    isOpen,
    onClose,
    onConfirm,
    title,
    message,
    confirmText = 'Confirm',
    cancelText = 'Cancel',
    type = 'confirm',
    inputValue = '',
    setInputValue = () => {},
    inputPlaceholder = '',
    inputLabel = '',
    showBlockOption = false,
    isBlockChecked = false,
    setBlockChecked = () => {},
}) {
    if (!isOpen) return null;

    const renderContent = () => {
        if (type === 'input') {
            return (
                <>
                    {inputLabel && <label>{inputLabel}</label>}
                    <input
                        type="text"
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        placeholder={inputPlaceholder}
                        autoFocus
                        onKeyPress={(e) => e.key === 'Enter' && onConfirm()}
                    />
                </>
            );
        }

        if (type === 'report') {
            return (
                <>
                    <p className="report-description">{message}</p>
                    
                    <div className="report-textarea-wrapper">
                        <label>Reason</label>
                        <textarea
                            value={inputValue}
                            onChange={(e) => setInputValue(e.target.value)}
                            placeholder="Describe the issue..."
                            rows={4}
                            autoFocus
                        />
                    </div>

                    {showBlockOption && (
                        <label className="report-block-option">
                            <div className="report-block-checkbox">
                                <input
                                    type="checkbox"
                                    checked={isBlockChecked}
                                    onChange={(e) => setBlockChecked(e.target.checked)}
                                />
                                <span>Block this user</span>
                            </div>
                            <span className="report-block-sub">They won't be able to message or call you.</span>
                        </label>
                    )}
                </>
            );
        }

        return <p>{message}</p>;
    };

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className={`modal-content ${type === 'report' ? 'modal-report' : ''}`} onClick={(e) => e.stopPropagation()}>
                <div className="modal-header">
                    <h3>{title}</h3>
                    <button className="modal-close" onClick={onClose}>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <line x1="18" y1="6" x2="6" y2="18"/>
                            <line x1="6" y1="6" x2="18" y2="18"/>
                        </svg>
                    </button>
                </div>

                <div className="modal-body">
                    {renderContent()}
                </div>

                <div className="modal-footer">
                    {cancelText && (
                        <button className="modal-btn modal-btn-cancel" onClick={onClose}>
                            {cancelText}
                        </button>
                    )}
                    <button 
                        className={`modal-btn modal-btn-confirm ${type === 'report' ? 'report' : ''}`} 
                        onClick={onConfirm}
                        disabled={type === 'report' && !inputValue.trim()}
                    >
                        {confirmText}
                    </button>
                </div>
            </div>
        </div>
    );
}