import { ArrowDownRight } from "lucide-react";

import { ProfileForm } from "@/components/onboarding/profile-form";
import { SiteFooter } from "@/components/site-footer";
import { Wordmark } from "@/components/ui/wordmark";

export default function OnboardingPage() {
  return (
    <main className="onboarding-page">
      <div className="onboarding-grid" aria-hidden="true" />
      <header className="onboarding-header">
        <Wordmark />
        <span className="edition-label">YOUR PERSONAL PROGRAM</span>
      </header>

      <section className="onboarding-hero">
        <div className="hero-copy">
          <p className="eyebrow"><span /> YOUR TASTE, CONTINUED</p>
          <h1>Find what to<br />watch <em>next.</em></h1>
          <p className="hero-lede">
            Turn your Letterboxd history into focused, personal film shelves that feel familiar enough to trust and surprising enough to explore.
          </p>
          <div className="hero-credibility">
            <ArrowDownRight aria-hidden="true" />
            <span>No account needed. Recommendations built from films you actually watched.</span>
          </div>
        </div>

        <div className="onboarding-card">
          <div className="card-index" aria-hidden="true">01 / START HERE</div>
          <ProfileForm />
        </div>

        <div className="poster-composition" aria-hidden="true">
          <div className="editorial-poster poster-one"><span>YOUR<br />HISTORY</span><i>01</i></div>
          <div className="editorial-poster poster-two"><span>TASTE<br />SIGNALS</span><i>02</i></div>
          <div className="editorial-poster poster-three"><span>WHAT’S<br />NEXT</span><i>03</i></div>
        </div>
      </section>

      <SiteFooter />
    </main>
  );
}
