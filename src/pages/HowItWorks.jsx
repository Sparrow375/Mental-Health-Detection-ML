import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import PageTransition from '../components/PageTransition';
import AnimatedSection from '../components/AnimatedSection';
import { steps } from '../data/steps';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/card';

export default function HowItWorks() {
  return (
    <PageTransition>
      {/* Header */}
      <section className="pt-32 pb-16 px-6 bg-primary">
        <div className="max-w-4xl mx-auto text-center">
          <AnimatedSection>
            <span className="text-sm font-semibold uppercase tracking-wider text-secondary">The Journey</span>
            <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold font-heading mt-3 mb-5 text-text-primary">
              How Lumen. <span className="text-secondary">Works</span>
            </h1>
            <p className="text-lg text-text-secondary max-w-2xl mx-auto leading-relaxed">
              From a single install to continuous, intelligent monitoring — here's how Lumen.
              turns your phone into a silent guardian for your mental wellbeing.
            </p>
          </AnimatedSection>
        </div>
      </section>

      {/* Timeline */}
      <section className="py-16 px-6 bg-primary">
        <div className="max-w-5xl mx-auto">
          <div className="relative">
            {/* Vertical line (desktop) */}
            <div className="hidden md:block absolute left-1/2 top-0 bottom-0 w-px bg-gold-border" />

            <div className="hidden md:block">
              {steps.map((step, i) => (
                <AnimatedSection key={i} delay={i * 0.2}>
                  <div className={`relative flex flex-col md:flex-row items-center gap-8 mb-16 last:mb-0 ${
                    i % 2 === 0 ? 'md:flex-row' : 'md:flex-row-reverse'
                  }`}>
                    {/* Content */}
                    <div className={`flex-1 ${i % 2 === 0 ? 'md:text-right md:pr-12' : 'md:text-left md:pl-12'}`}>
                      <Card className="hover:shadow-xl hover:shadow-secondary/10 transition-all duration-500">
                        <CardHeader>
                           <span className="text-5xl font-extrabold text-secondary opacity-30">{step.number}</span>
                           <CardTitle className="text-2xl mt-2 mb-1">{step.title}</CardTitle>
                        </CardHeader>
                        <CardContent>
                           <CardDescription>{step.description}</CardDescription>
                        </CardContent>
                      </Card>
                    </div>

                    {/* Center node - Square for industrial look */}
                    <div className="hidden md:flex absolute left-1/2 -translate-x-1/2 w-12 h-12 bg-primary border border-secondary items-center justify-center text-secondary z-10">
                      {step.icon}
                    </div>

                    {/* Spacer */}
                    <div className="flex-1 hidden md:block" />
                  </div>
                </AnimatedSection>
              ))}
            </div>
          </div>

          {/* Mobile timeline */}
          <div className="md:hidden space-y-8">
            {steps.map((step, i) => (
              <AnimatedSection key={i} delay={i * 0.15}>
                <div className="flex gap-4">
                  <div className="flex flex-col items-center">
                    <div className="w-10 h-10 bg-primary border border-secondary flex items-center justify-center text-secondary flex-shrink-0">
                      {step.icon}
                    </div>
                    {i < steps.length - 1 && (
                      <div className="w-px flex-1 bg-gold-border mt-2" />
                    )}
                  </div>
                  <Card className="flex-1">
                    <CardHeader className="pb-2">
                       <span className="text-xs font-bold text-secondary uppercase tracking-wider">Step {step.number}</span>
                       <CardTitle className="text-lg mt-1">{step.title}</CardTitle>
                    </CardHeader>
                    <CardContent>
                       <CardDescription>{step.description}</CardDescription>
                    </CardContent>
                  </Card>
                </div>
              </AnimatedSection>
            ))}
          </div>
        </div>
      </section>

      {/* Privacy Callout */}
      <section className="py-24 px-6 bg-primary border-t border-gold-border/20">
        <div className="max-w-4xl mx-auto">
          <AnimatedSection>
            <div className="relative overflow-hidden border border-gold-border bg-primary-light p-10 sm:p-16 text-center">
              <div className="relative z-10">
                <div className="w-16 h-16 border border-secondary flex items-center justify-center mx-auto mb-6 text-secondary">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-8 h-8">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 10.5V6.75a4.5 4.5 0 10-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H6.75a2.25 2.25 0 00-2.25 2.25v6.75a2.25 2.25 0 002.25 2.25z" />
                  </svg>
                </div>
                <h3 className="text-3xl sm:text-4xl font-bold font-heading text-text-primary mb-4">
                  Your Data Stays With You
                </h3>
                <p className="text-text-secondary max-w-xl mx-auto leading-relaxed text-lg">
                  Every calculation happens on your device. No data is uploaded to the cloud.
                  No third-party has access. Lumen. is designed so that even we can't see your data.
                </p>
              </div>
            </div>
          </AnimatedSection>
        </div>
      </section>

      {/* CTA */}
      <section className="py-16 pb-40 px-6 bg-primary">
        <AnimatedSection className="text-center">
          <h2 className="text-2xl sm:text-3xl font-bold font-heading text-text-primary mb-4">
            Built for the people who need it most
          </h2>
          <p className="text-text-secondary mb-8 max-w-lg mx-auto">
            Early detection can change everything. Lumen. makes it passive, personal, and private.
          </p>
          <Link
            to="/team"
            className="inline-block px-8 py-3.5 bg-secondary text-primary font-semibold text-sm hover:shadow-lg hover:shadow-secondary/25 transition-all duration-300"
          >
            Meet the Team
          </Link>
        </AnimatedSection>
      </section>
    </PageTransition>
  );
}
