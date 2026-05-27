import { motion } from 'framer-motion';
import PageTransition from '../components/PageTransition';
import AnimatedSection from '../components/AnimatedSection';
import screenshot from '../../images/Screenshot_20260420_010628_Swasthiti.jpg';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/card';

const features = [
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
        <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 14.25v2.25m3-4.5v4.5m3-6.75v6.75m3-9v9M6 20.25h12A2.25 2.25 0 0020.25 18V6A2.25 2.25 0 0018 3.75H6A2.25 2.25 0 003.75 6v12A2.25 2.25 0 006 20.25z" />
      </svg>
    ),
    title: 'Passive Data Collection',
    desc: 'Monitors 29 behavioral features across sleep, mobility, social, and usage patterns — all without any user action.',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
        <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 1.5H8.25A2.25 2.25 0 006 3.75v16.5a2.25 2.25 0 002.25 2.25h7.5A2.25 2.25 0 0018 20.25V3.75a2.25 2.25 0 00-2.25-2.25H13.5m-3 0V3h3V1.5m-3 0h3m-3 18.75h3" />
      </svg>
    ),
    title: 'On-Device Processing',
    desc: 'All analysis runs locally on your phone. Raw behavioral data is never uploaded, transmitted, or shared with anyone.',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
        <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
      </svg>
    ),
    title: 'Personalized Baselines',
    desc: 'Learns what normal looks like for you — not a population average. Your 28-day baseline is uniquely yours.',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    title: 'Early Detection',
    desc: 'Detects sustained behavioral shifts before they become severe — giving people time to seek support on their terms.',
  },
];

export default function Demo() {
  return (
    <PageTransition>
      {/* Header */}
      <section className="pt-32 pb-16 px-6 bg-primary">
        <div className="max-w-4xl mx-auto text-center">
          <AnimatedSection>
            <span className="text-sm font-semibold uppercase tracking-wider text-secondary">Preview</span>
            <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold font-heading mt-3 mb-5 text-text-primary">
              See Lumen. <span className="text-secondary">In Action</span>
            </h1>
            <p className="text-lg text-text-secondary max-w-2xl mx-auto leading-relaxed">
              A glimpse into the interface and experience of using Lumen.
            </p>
          </AnimatedSection>
        </div>
      </section>

      {/* Phone Mockup */}
      <section className="py-8 px-6 bg-primary">
        <div className="max-w-md mx-auto">
          <AnimatedSection>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.8 }}
              className="relative mx-auto border border-gold-border p-2 bg-primary-light"
            >
              {/* Phone frame - sharp instead of rounded */}
              <div className="relative w-full aspect-[9/19] bg-primary overflow-hidden">
                <div className="w-full h-full bg-primary-dark overflow-hidden relative">
                  {/* Screenshot Content */}
                  <div className="w-full h-full overflow-hidden">
                    <img 
                      src={screenshot} 
                      alt="Lumen. App Screenshot" 
                      className="w-full h-full object-cover"
                    />
                  </div>
                </div>
              </div>
            </motion.div>
          </AnimatedSection>
        </div>
      </section>

      {/* Feature Grid */}
      <section className="py-24 px-6 bg-primary border-t border-gold-border/20">
        <div className="max-w-5xl mx-auto">
          <AnimatedSection className="text-center mb-16">
            <span className="text-sm font-semibold uppercase tracking-wider text-secondary">Core Features</span>
            <h2 className="text-3xl sm:text-4xl font-bold font-heading mt-3 text-text-primary">
              What Makes Lumen. Different
            </h2>
          </AnimatedSection>

          <div className="grid sm:grid-cols-2 gap-6">
            {features.map((feature, i) => (
              <AnimatedSection key={i} delay={i * 0.15}>
                <Card className="h-full hover:shadow-xl hover:shadow-secondary/10 transition-all duration-500">
                  <CardHeader>
                    <div className="w-12 h-12 bg-primary-light border border-secondary flex items-center justify-center text-secondary mb-5">
                      {feature.icon}
                    </div>
                    <CardTitle>{feature.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CardDescription>{feature.desc}</CardDescription>
                  </CardContent>
                </Card>
              </AnimatedSection>
            ))}
          </div>
        </div>
      </section>

      {/* Coming Soon */}
      <section className="py-16 pb-40 px-6 bg-primary">
        <AnimatedSection>
          <Card className="max-w-2xl mx-auto text-center border-gold-border p-6">
            <CardHeader className="flex flex-col items-center">
              <div className="w-16 h-16 border border-secondary bg-primary-light flex items-center justify-center mb-6 text-secondary">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-8 h-8">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M21 7.5l-2.25-1.313M21 7.5v2.25m0-2.25l-2.25 1.313M3 7.5l2.25-1.313M3 7.5l2.25 1.313M3 7.5v2.25m9 3l2.25-1.313M12 12.75l-2.25-1.313M12 12.75V15m0 6.75l2.25-1.313M12 21.75V19.5m0 2.25l-2.25-1.313m0-16.875L12 2.25l2.25 1.313M21 14.25v2.25l-2.25 1.313m-13.5 0L3 16.5v-2.25" />
                </svg>
              </div>
              <CardTitle className="text-2xl mb-3">Coming Soon</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription className="leading-relaxed">
                Lumen. is currently in the research and development phase. We're working
                to bring passive, personalized mental health screening to everyone.
              </CardDescription>
            </CardContent>
          </Card>
        </AnimatedSection>
      </section>
    </PageTransition>
  );
}
