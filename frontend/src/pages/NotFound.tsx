
import { useLocation } from "react-router-dom";
import { useEffect } from "react";
import { Link } from "react-router-dom";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error(
      "404 Error: User attempted to access non-existent route:",
      location.pathname
    );
  }, [location.pathname]);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-police-background text-police-foreground p-4">
      <div className="w-full max-w-md text-center space-y-6">
        <div className="p-4 rounded-full bg-blue-900/20 inline-flex">
          <div className="text-blue-500 text-6xl">404</div>
        </div>
        <h1 className="text-4xl font-bold text-white">Page not found</h1>
        <p className="text-xl text-gray-400">The page you're looking for doesn't exist or has been moved.</p>
        <Link to="/" className="inline-block px-8 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white rounded-lg font-medium shadow-lg shadow-blue-900/30 border border-blue-700/50 transition-all duration-200 hover:scale-105">
          Return to Home
        </Link>
      </div>
    </div>
  );
};

export default NotFound;
