
import React from 'react';
import LandingHero from '../components/LandingHero';
import Footer from '../components/Footer';

const Index: React.FC = () => {
  return (
    <div className="min-h-screen flex flex-col bg-police-background text-police-foreground">
      <LandingHero />
      <Footer />
    </div>
  );
};

export default Index;
