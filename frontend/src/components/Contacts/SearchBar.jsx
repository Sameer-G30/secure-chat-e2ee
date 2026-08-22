// src/components/Contacts/SearchBar.jsx
export default function SearchBar({
    searchTerm,
    setSearchTerm,
    handleSearchUsers,
    isSearching
}) {
    return (
        <div className="search-bar">
            <div className="search-input-wrapper">
                <svg className="search-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="11" cy="11" r="8"/>
                    <path d="M21 21l-4.35-4.35"/>
                </svg>
                <input 
                    type="text" 
                    placeholder="Search users..." 
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSearchUsers()}
                />
            </div>
            <button 
                className="search-btn" 
                onClick={handleSearchUsers}
                disabled={isSearching}
            >
                {isSearching ? '...' : 'Search'}
            </button>
        </div>
    );
}