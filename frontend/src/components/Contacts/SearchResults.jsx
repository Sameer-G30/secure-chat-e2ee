// src/components/Contacts/SearchResults.jsx
export default function SearchResults({ searchResults, handleAddContact }) {
    if (searchResults.length === 0) return null;

    return (
        <div className="search-results">
            <div className="search-results-header">Search Results</div>
            {searchResults.map((user) => (
                <div key={user.uid} className="search-result-item">
                    <div className="search-result-info">
                        <div className="search-result-avatar">
                            {user.username.charAt(0).toUpperCase()}
                        </div>
                        <div>
                            <div className="search-result-name">{user.username}</div>
                            <div className="search-result-email">{user.email}</div>
                        </div>
                    </div>
                    <button 
                        className="add-contact-btn"
                        onClick={() => handleAddContact(user.uid, user.username)}
                    >
                        Add
                    </button>
                </div>
            ))}
        </div>
    );
}