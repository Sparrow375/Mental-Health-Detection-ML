import { motion } from 'framer-motion';
import PageTransition from '../components/PageTransition';
import AnimatedSection from '../components/AnimatedSection';
import { teamMembers } from '../data/team';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/card';

export default function Team() {
  return (
    <PageTransition>
      {/* Header */}
      <section className="pt-32 pb-16 px-6 bg-primary">
        <div className="max-w-4xl mx-auto text-center">
          <AnimatedSection>
            <span className="text-sm font-semibold uppercase tracking-wider text-secondary">The People</span>
            <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold font-heading mt-3 mb-5 text-text-primary">
              Meet the <span className="text-secondary">Team</span>
            </h1>
            <p className="text-lg text-text-secondary max-w-2xl mx-auto leading-relaxed">
              A small team with a shared belief: mental health detection should be
              passive, personal, and accessible to everyone.
            </p>
          </AnimatedSection>
        </div>
      </section>

      {/* Team Grid */}
      <section className="py-16 px-6 bg-primary">
        <div className="max-w-5xl mx-auto">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {teamMembers.map((member, i) => (
              <AnimatedSection key={i} delay={i * 0.15}>
                <Card className="text-center h-full hover:shadow-xl hover:shadow-secondary/10 transition-shadow duration-500">
                  <CardHeader className="flex flex-col items-center">
                    <div className="w-20 h-20 bg-primary-light border border-secondary flex items-center justify-center mb-5 shadow-sm">
                      <span className="text-2xl font-bold text-secondary">{member.initials}</span>
                    </div>
                    <CardTitle className="mb-1">{member.name}</CardTitle>
                    <CardDescription>{member.role}</CardDescription>
                  </CardHeader>
                </Card>
              </AnimatedSection>
            ))}
          </div>
        </div>
      </section>

      {/* Mission */}
      <section className="py-24 px-6 bg-primary pb-40 border-t border-gold-border/20">
        <div className="max-w-4xl mx-auto">
          <AnimatedSection>
            <div className="relative overflow-hidden border border-gold-border bg-primary p-10 sm:p-16">
              <div className="relative z-10 text-center">
                <h3 className="text-3xl sm:text-4xl font-bold font-heading text-secondary mb-6">Our Mission</h3>
                <p className="text-text-secondary text-lg leading-relaxed max-w-2xl mx-auto">
                  We believe that the most meaningful health insights come not from comparing people to populations,
                  but from understanding how each person's own patterns change over time. Lumen. exists to make
                  that understanding possible — silently, privately, and with the respect that every individual deserves.
                </p>
              </div>
            </div>
          </AnimatedSection>
        </div>
      </section>
    </PageTransition>
  );
}
