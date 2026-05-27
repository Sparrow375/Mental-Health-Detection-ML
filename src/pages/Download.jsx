import { motion } from 'framer-motion';
import PageTransition from '../components/PageTransition';
import AnimatedSection from '../components/AnimatedSection';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/card';

export default function Download() {
  return (
    <PageTransition>
      <section className="pt-32 pb-40 px-6 min-h-screen bg-primary">
        <div className="max-w-4xl mx-auto">
          <AnimatedSection className="text-center mb-16">
            <span className="text-sm font-semibold uppercase tracking-wider text-secondary">Get Involved</span>
            <h1 className="text-4xl md:text-5xl font-bold font-heading mt-3 mb-6 text-text-primary">
              Beta Testing & Downloads
            </h1>
            <p className="text-text-secondary max-w-2xl mx-auto text-lg leading-relaxed">
              Help us refine Lumen. by participating in our beta program. Download the latest builds for Android and provide valuable feedback.
            </p>
          </AnimatedSection>

          <div className="grid md:grid-cols-2 gap-8 mb-16">
            {/* User Build */}
            <AnimatedSection delay={0.1}>
              <Card className="h-full flex flex-col">
                <CardHeader>
                  <div className="w-14 h-14 border border-secondary bg-primary flex items-center justify-center mb-6 text-secondary">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-7 h-7">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M12 21v-8.25M15.75 21h-7.586a2.25 2.25 0 0 1-1.591-.659l-3.152-3.152a2.25 2.25 0 0 1 1.13-3.845b4.5 4.5 0 0 1 2.427.592l.848.424a1.5 1.5 0 0 0 1.341.01l.492-.246m14.016-4.992a2.25 2.25 0 0 0-3.181 3.182m0-4.991v3.033c0 .114-.045.223-.125.304l-3.033 3.033" />
                      <path strokeLinecap="round" strokeLinejoin="round" d="M15 11.25A2.25 2.25 0 0 0 12.75 9h-1.5a2.25 2.25 0 0 0-2.25 2.25v1.5a2.25 2.25 0 0 0 2.25 2.25h1.5a2.25 2.25 0 0 0 2.25-2.25v-1.5Z" />
                    </svg>
                  </div>
                  <CardTitle className="text-2xl mb-2">User Build</CardTitle>
                </CardHeader>
                <CardContent className="flex-grow flex flex-col">
                  <CardDescription className="mb-8 flex-grow">
                    The standard Lumen. experience. Recommended for most beta testers wanting to experience the app as intended.
                  </CardDescription>
                  <a
                    href="/lumen_user_build.zip"
                    download="Lumen user build.zip"
                    className="inline-flex items-center justify-center gap-2 px-6 py-3.5 bg-secondary text-primary font-semibold text-sm w-full hover:bg-secondary/90 transition-colors"
                  >
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-5 h-5">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
                    </svg>
                    Download User APK
                  </a>
                </CardContent>
              </Card>
            </AnimatedSection>

            {/* Dev Build */}
            <AnimatedSection delay={0.2}>
              <Card className="h-full flex flex-col">
                <CardHeader>
                  <div className="w-14 h-14 border border-secondary bg-primary flex items-center justify-center mb-6 text-secondary">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-7 h-7">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M11.42 15.17L17.25 21A2.652 2.652 0 0021 17.25l-5.877-5.877M11.42 15.17l2.496-3.03c.317-.384.74-.626 1.208-.766M11.42 15.17l-4.655 5.653a2.548 2.548 0 11-3.586-3.586l6.837-5.63m5.108-.233c.55-.164 1.163-.188 1.743-.14a4.5 4.5 0 004.486-6.336l-3.276 3.277a3.004 3.004 0 01-2.25-2.25l3.276-3.276a4.5 4.5 0 00-6.336 4.486c.091 1.076-.071 2.264-.904 2.95l-.102.085m-1.745 1.437L5.909 7.5H4.5L2.25 3.75l1.5-1.5L7.5 4.5v1.409l4.26 4.26m-1.745 1.437l1.745-1.437m6.615 8.206L15.75 15.75M4.867 19.125h.008v.008h-.008v-.008z" />
                    </svg>
                  </div>
                  <CardTitle className="text-2xl mb-2">Dev Build</CardTitle>
                </CardHeader>
                <CardContent className="flex-grow flex flex-col">
                  <CardDescription className="mb-8 flex-grow">
                    Includes experimental features, debug menus, and extra logging. Best for developers and advanced testers.
                  </CardDescription>
                  <a
                    href="/lumen_dev_build.zip"
                    download
                    className="inline-flex items-center justify-center gap-2 px-6 py-3.5 border border-secondary text-secondary font-semibold text-sm w-full hover:bg-secondary/10 transition-colors"
                  >
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-5 h-5">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
                    </svg>
                    Download Dev APK
                  </a>
                </CardContent>
              </Card>
            </AnimatedSection>
          </div>

          {/* Instructions */}
          <AnimatedSection delay={0.3}>
            <Card>
              <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 border-b border-gold-border/20 p-6">
                <div className="w-12 h-12 border border-secondary bg-primary flex items-center justify-center text-secondary flex-shrink-0">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-6 h-6">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z" />
                  </svg>
                </div>
                <div>
                  <CardTitle className="text-xl">Installation Instructions</CardTitle>
                  <CardDescription>How to install the APK on your Android device</CardDescription>
                </div>
              </div>

              <CardContent className="pt-6">
                <div className="space-y-8">
                  {[
                    { title: "Download the ZIP", desc: "Click the download button above to get the ZIP file containing the APK." },
                    { title: "Extract the APK", desc: "Locate the ZIP file in your \"Downloads\" folder, tap on it, and select \"Extract\" to retrieve the APK file." },
                    { title: "Open the APK", desc: "Tap on the extracted APK file to begin the installation process." },
                    { title: "Allow Unknown Apps", desc: "If prompted, you'll need to allow your file manager or browser to install unknown apps. Tap \"Settings\" on the prompt and toggle on \"Allow from this source\"." },
                    { title: "Bypass Play Protect", desc: "Google Play Protect may show a warning. Tap \"More details\" and then \"Install anyway\" to complete the setup." }
                  ].map((step, i) => (
                    <div key={i} className="flex gap-5">
                      <div className="flex-shrink-0 w-8 h-8 border border-secondary/50 text-secondary flex items-center justify-center font-bold text-sm bg-primary-light">
                        {i + 1}
                      </div>
                      <div>
                        <h4 className="text-text-primary font-bold mb-1">{step.title}</h4>
                        <p className="text-text-secondary text-sm leading-relaxed">{step.desc}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </AnimatedSection>
        </div>
      </section>
    </PageTransition>
  );
}
