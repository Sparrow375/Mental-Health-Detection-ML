import { useRef, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import PageTransition from '../components/PageTransition';
import AnimatedSection from '../components/AnimatedSection';
import { stats } from '../data/stats';
import { LampContainer } from '../components/ui/lamp';
import { BentoGrid, BentoGridItem } from '../components/ui/bento-grid';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/card';
import { Activity, Shield, Cpu, Target } from 'lucide-react';

function CountUp({ target, suffix, duration = 2 }) {
  const [count, setCount] = useState(0);
  const ref = useRef(null);
  const [started, setStarted] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting && !started) setStarted(true); },
      { threshold: 0.5 }
    );
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, [started]);

  useEffect(() => {
    if (!started) return;
    let frame;
    const start = performance.now();
    const step = (now) => {
      const progress = Math.min((now - start) / (duration * 1000), 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setCount(Math.round(eased * target));
      if (progress < 1) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [started, target, duration]);

  return (
    <span ref={ref} className="tabular-nums">
      {count}{suffix}
    </span>
  );
}

export default function Home() {
  return (
    <PageTransition>
      {/* Hero with Lamp Effect */}
      <LampContainer>
        <motion.h1
          initial={{ opacity: 0.5, y: 100 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{
            delay: 0.3,
            duration: 0.8,
            ease: "easeInOut",
          }}
          className="mt-8 bg-gradient-to-br from-text-primary to-text-secondary py-4 bg-clip-text text-center text-4xl font-medium tracking-tight text-transparent md:text-7xl font-heading"
        >
          Lumen.
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.8 }}
          className="text-text-secondary text-lg md:text-xl max-w-2xl text-center mt-4 mb-8 font-sans"
        >
          Lumen. learns your personal behavioral baseline and silently monitors for sustained changes
          that could signal early mental health risks — all on your phone, all private.
        </motion.p>
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ delay: 0.7, duration: 0.8 }}
          className="flex flex-col sm:flex-row gap-4 justify-center mt-4"
        >
          <Link
            to="/how-it-works"
            className="px-8 py-3 bg-secondary text-primary font-semibold text-sm hover:bg-secondary-light transition-colors"
          >
            See How It Works
          </Link>
          <Link
            to="/demo"
            className="px-8 py-3 border border-gold-border text-text-primary font-semibold text-sm hover:bg-glass transition-colors"
          >
            View Demo
          </Link>
        </motion.div>
      </LampContainer>

      {/* Problem */}
      <section className="py-24 sm:py-32 px-6 bg-primary">
        <div className="max-w-5xl mx-auto">
          <AnimatedSection className="text-center mb-16">
            <span className="text-sm font-semibold uppercase tracking-wider text-secondary">The Problem</span>
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold font-heading mt-3 mb-4 text-text-primary">
              The Silent Crisis
            </h2>
            <p className="text-text-secondary max-w-2xl mx-auto text-lg">
              Mental health conditions are the leading cause of disability worldwide. The vast majority go
              undetected until symptoms become severe.
            </p>
          </AnimatedSection>

          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                stat: '970M+',
                title: 'People Affected',
                desc: 'Mental health disorders affect nearly 1 billion people globally, with depression and anxiety leading the count.',
              },
              {
                stat: '75%+',
                title: 'Never Treated',
                desc: 'In low and middle-income countries, over 75% of people with mental health conditions receive no treatment at all.',
              },
              {
                stat: 'Too Late',
                title: 'Late Detection',
                desc: 'Most individuals don\'t recognize their own behavioral changes as clinically significant until conditions have progressed.',
              },
            ].map((card, i) => (
              <AnimatedSection key={i} delay={i * 0.15}>
                <Card className="h-full">
                  <CardHeader>
                    <div className="text-4xl font-bold text-secondary mb-3">{card.stat}</div>
                    <CardTitle className="text-text-primary">{card.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CardDescription>{card.desc}</CardDescription>
                  </CardContent>
                </Card>
              </AnimatedSection>
            ))}
          </div>
        </div>
      </section>

      {/* Solution - Bento Grid */}
      <section className="py-24 sm:py-32 px-6 bg-primary border-t border-gold-border/20">
        <div className="max-w-5xl mx-auto">
          <AnimatedSection className="text-center mb-16">
            <span className="text-sm font-semibold uppercase tracking-wider text-secondary">Our Approach</span>
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold font-heading mt-3 mb-4 text-text-primary">
              A New Paradigm
            </h2>
            <p className="text-text-secondary max-w-2xl mx-auto text-lg">
              Instead of comparing you to everyone else, Lumen. compares you to yourself.
            </p>
          </AnimatedSection>

          <BentoGrid>
            <BentoGridItem
              title="Personalized Baseline"
              description="Lumen. builds a unique behavioral baseline for each person — learning what normal looks like for you, not the population."
              header={<div className="h-32 w-full bg-primary-light flex items-center justify-center text-text-secondary"><Activity size={48} strokeWidth={1}/></div>}
              icon={<Target className="w-5 h-5" />}
              className="md:col-span-2"
            />
            <BentoGridItem
              title="Passive Collection"
              description="No mood logs. No daily check-ins. No interruptions. Silently collects data through sensors."
              header={<div className="h-32 w-full bg-primary-light flex items-center justify-center text-text-secondary"><Cpu size={48} strokeWidth={1}/></div>}
              icon={<Activity className="w-5 h-5" />}
              className="md:col-span-1"
            />
            <BentoGridItem
              title="Private Processing"
              description="All processing happens on your device. Raw behavioral data never leaves your phone. No cloud. No third-party access."
              header={<div className="h-32 w-full bg-primary-light flex items-center justify-center text-text-secondary"><Shield size={48} strokeWidth={1}/></div>}
              icon={<Shield className="w-5 h-5" />}
              className="md:col-span-3"
            />
          </BentoGrid>
        </div>
      </section>

      {/* Stats */}
      <section className="py-24 sm:py-32 px-6 bg-primary border-t border-gold-border/20 pb-40">
        <div className="max-w-5xl mx-auto">
          <AnimatedSection className="text-center mb-16">
            <span className="text-sm font-semibold uppercase tracking-wider text-secondary">By The Numbers</span>
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold font-heading mt-3 text-text-primary">
              The Scale of Impact
            </h2>
          </AnimatedSection>

          <div className="grid grid-cols-2 md:grid-cols-5 gap-6">
            {stats.map((stat, i) => (
              <AnimatedSection key={i} delay={i * 0.1}>
                <div className="rounded-2xl overflow-hidden border border-gold-border bg-primary p-6 text-center h-full flex flex-col justify-center">
                  <span className="text-3xl sm:text-4xl font-bold text-secondary block mb-2">
                    <CountUp target={stat.value} suffix={stat.suffix} />
                  </span>
                  <span className="text-xs text-text-secondary leading-relaxed">{stat.label}</span>
                </div>
              </AnimatedSection>
            ))}
          </div>
        </div>
      </section>
    </PageTransition>
  );
}
