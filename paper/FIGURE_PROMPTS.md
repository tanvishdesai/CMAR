# Figure and Diagram Prompts

The LaTeX draft now includes generated figure assets from `paper/figures/`. The prompts below are kept as regeneration instructions for future revisions. Each figure should be exported as PDF or high-resolution PNG with a white background, readable labels, and no decorative gradients.

## Figure 1: CertAV architecture and certification workflow

Prompt:

Create a clean IEEE conference-style technical diagram for a method named "CertAV". Show an input audio-video clip on the left. The video branch samples 16 frames and passes them through a frozen DINOv2 visual encoder, producing temporal visual features. The audio branch passes waveform audio through a frozen Whisper encoder, producing temporal audio features. Show both branches entering temporal aggregation modules, then a bidirectional cross-modal attention fusion block, then a binary real/fake classifier. Around the frozen feature tensors, show Gaussian feature-space noise being injected during smoothing. On the right, show Monte Carlo votes, a Clopper-Pearson lower confidence bound, and the certified radius formula R = sigma Phi^{-1}(lower p_A). Use restrained colors: dark gray text, blue for visual features, green for audio features, amber for noise/certification. Use flat vector shapes, thin arrows, and concise labels. Do not use cartoons, stock photos, or 3D effects.

## Figure 2: Certified accuracy versus radius

Prompt:

Create a publication-quality line plot titled "Certified accuracy on FakeAVCeleb". The x-axis is certified radius r in joint feature-space L2 units. The y-axis is certified accuracy (%). Plot four curves for sigma values 0.12, 0.25, 0.50, and 1.00. Use the following anchor values: sigma 0.12: 88.8% at r=0.25 and 0% at r=0.50; sigma 0.25: 89.4% at r=0.25, 86.6% at r=0.50, 0% at r=1.00; sigma 0.50: 90.5% at r=0.25, 89.2% at r=0.50, 85.4% at r=1.00, 0% at r=1.50; sigma 1.00: 91.8% at r=0.25, 90.7% at r=0.50, 88.0% at r=1.00, 84.9% at r=1.50. Include small error bands where available, a compact legend, and a grid with light gray lines. Style should match an IEEE ICASSP paper figure.

## Figure 3: Representation manifold and smoothing intuition

Prompt:

Create a two-panel technical figure explaining why feature-space smoothing works for CertAV. Panel A shows a PCA cumulative variance plot for visual, audio, and joint features. Mark the 90% and 95% variance thresholds. Use these dimensions: visual d90=73, d95=113 out of 384; audio d90=13, d95=36 out of 384; joint d90=75, d95=116 out of 768. Panel B is a schematic: a low-dimensional data manifold embedded in a high-dimensional feature space, a classifier boundary staying away from most samples, and circular Gaussian smoothing neighborhoods around samples. Annotate "low intrinsic dimension", "large certified radius", and "1.0% flip rate at sigma=1.0". Use scientific, minimal styling with no decorative background.

## Figure 4: Robustness evaluation card for av-robustbench

Prompt:

Create a compact benchmark workflow diagram for "av-robustbench". Show a detector adapter feeding into four evaluation modules: adversarial attacks, randomized smoothing certification, common media degradations, and transfer tests. The modules output a robustness card and leaderboard JSON. Include attack labels "PGD-Linf", "PGD-L2", "Square", and "AutoAttackAV"; certification labels "n0=100", "n=1000", "alpha=0.001"; degradation labels "JPEG", "H.264", "social media". Make the diagram look like a reproducibility pipeline rather than marketing material. Use simple boxes, arrows, and IEEE-style typography.
