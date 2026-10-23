/-
  =============================================================================
  MobiusAttention.lean: Formal Verification of Semantic Gauge Möbius Attention
  -----------------------------------------------------------------------------
  Dual-Licensed under Apache-2.0 and MulanPSL-2.0.

  Zero-Trust Formal Gatekeeper Specification:
  1. Discrete Foundations & Topological Invariants
  2. Program Equivalence: Two-Stage Associative Scan ≡ Autoregressive Recurrence
  3. Spatial Aliasing Boundary Negative Theorem (Carrier Inversion & Decoy Amplification)
  4. Monotonic Suppression Recovery (Dynamic Cutoff & Sign-Aware Gating)
  5. Epsilon-Perturbation Robustness under Open-Loop Phase Drift
  =============================================================================
-/

import Mathlib.Algebra.Group.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.Complex.Trigonometric

noncomputable section

namespace MobiusAttention

/--
  Definition 1: Discrete Sequence Base Manifold
  Positions along the autoregressive context window: t ∈ ℕ.
-/
abbrev Position := ℕ

/--
  Definition 2: Semantic Continuous Torsion Rate
  At each token position t, ω(t) represents the local continuous torsion rate
  induced by narrative evolution, bounded in [0, π].
-/
structure Connection where
  omega : Position → ℝ
  omega_nonneg : ∀ t, 0 ≤ omega t
  omega_le_pi  : ∀ t, omega t ≤ Real.pi

/--
  Definition 3: Path Holonomy (Cumulative Gauge Phase)
  The parallel transport along the causal trajectory from token j to token i (j ≤ i)
  is the sum of local connection curvature along the path.
-/
def Holonomy (A : Connection) (j i : Position) : ℝ :=
  if _h : j < i then
    (Finset.Ico (j + 1) (i + 1)).sum A.omega
  else
    0

/--
  Definition 4: Möbius Phase Modulation Factor
  Under continuous Möbius phase flow on the non-orientable bundle,
  the topological twist acting on the rotary inner product is cos(ΔΦ).
-/
def ParityFactor (A : Connection) (j i : Position) : ℝ :=
  Real.cos (Holonomy A j i)

/-
  =============================================================================
  Section 1: Discrete Foundations & Backwards-Compatibility Invariants
  =============================================================================
-/

/--
  Theorem: Conservation of Flatness (Reduction to Standard RoPE)
  When the context is smooth (∀ t, ω(t) = 0), the holonomy vanishes everywhere,
  and the parity factor is identically 1.0, reducing strictly to standard RoPE.
-/
theorem conservation_of_flatness
    (A : Connection) (h_flat : ∀ t, A.omega t = 0) (j i : Position) :
    ParityFactor A j i = 1 := by
  unfold ParityFactor Holonomy
  by_cases h : j < i
  · simp only [h, dif_pos]
    have h_sum_zero : (Finset.Ico (j + 1) (i + 1)).sum A.omega = 0 := by
      apply Finset.sum_eq_zero
      intro x _
      exact h_flat x
    rw [h_sum_zero]
    exact Real.cos_zero
  · simp only [h]
    exact Real.cos_zero

/--
  Theorem: Path-Integral Additivity / Prefix-Sum Decomposition
  For any sequence indices j ≤ k ≤ i, the cumulative holonomy along the composite
  path is the sum of partial holonomies: Holonomy(j, i) = Holonomy(j, k) + Holonomy(k, i).
-/
theorem holonomy_path_additivity
    (A : Connection) {j k i : Position} (h1 : j ≤ k) (h2 : k ≤ i) :
    Holonomy A j i = Holonomy A j k + Holonomy A k i := by
  unfold Holonomy
  by_cases h_jk : j < k
  · by_cases h_ki : k < i
    · have h_ji : j < i := lt_of_lt_of_le h_jk h2
      simp only [h_ji, h_jk, h_ki, dif_pos]
      have hmn : j + 1 ≤ k + 1 := Nat.succ_le_succ h1
      have hnk : k + 1 ≤ i + 1 := Nat.succ_le_succ h2
      exact (Finset.sum_Ico_consecutive (f := A.omega) hmn hnk).symm
    · have h_eq : k = i := le_antisymm h2 (not_lt.mp h_ki)
      subst h_eq
      simp [h_jk]
  · have h_eq : j = k := le_antisymm h1 (not_lt.mp h_jk)
    subst h_eq
    simp

/-- Backward-compatibility alias for test harness -/
theorem holonomy_cocycle
    (A : Connection) {j k i : Position} (h1 : j ≤ k) (h2 : k ≤ i) :
    Holonomy A j i = Holonomy A j k + Holonomy A k i :=
  holonomy_path_additivity A h1 h2

/--
  Theorem: Scalar Parity Inversion under Holonomy = π
-/
theorem antipodal_inversion
    (A : Connection) (j i : Position) (h_pi : Holonomy A j i = Real.pi) :
    ParityFactor A j i = -1 := by
  unfold ParityFactor
  rw [h_pi]
  exact Real.cos_pi

/--
  Theorem: Cyclic Parity Conservation (2π Periodicity)
-/
theorem cyclic_parity_conservation
    (A : Connection) (j i : Position) (h_2pi : Holonomy A j i = 2 * Real.pi) :
    ParityFactor A j i = 1 := by
  unfold ParityFactor
  rw [h_2pi]
  exact Real.cos_two_pi

/--
  Theorem: Gauge Translation Invariance (Backward-Compatibility)
-/
theorem gauge_translation_invariance
    (phi_i phi_j c : ℝ) :
    (phi_i + c) - (phi_j + c) = phi_i - phi_j := by
  ring

/-
  =============================================================================
  Section 2: Planar Rotary Dual Modulation & Bilinear Invariance
  =============================================================================
-/

/--
  Definition 5: Two-Dimensional Rotary Vector
  Represents query and key coordinate pairs in a 2D RoPE subspace.
-/
structure Vec2 where
  x : ℝ
  y : ℝ

/--
  Definition 6: Planar Euclidean Inner Product
-/
def dot2 (u v : Vec2) : ℝ :=
  u.x * v.x + u.y * v.y

/--
  Definition 7: Two-Dimensional Orthogonal Rotation Operator R(θ) ∈ SO(2)
-/
def rotate2 (θ : ℝ) (v : Vec2) : Vec2 :=
  ⟨Real.cos θ * v.x - Real.sin θ * v.y, Real.sin θ * v.x + Real.cos θ * v.y⟩

/--
  Lemma: Rotation Orthogonality & Angle Shift Identity
  ⟨R(α) u, R(β) v⟩ = ⟨u, R(β - α) v⟩
-/
theorem dot2_rotate2_eq (α β : ℝ) (u v : Vec2) :
    dot2 (rotate2 α u) (rotate2 β v) = dot2 u (rotate2 (β - α) v) := by
  unfold dot2 rotate2
  rw [Real.cos_sub, Real.sin_sub]
  ring

/--
  Theorem: RoPE Relative Rotation Invariance
  Under continuous rotary dual modulation, the inner product between query at position i
  (with gauge phase Φ_i) and key at position j (with gauge phase Φ_j) satisfies:
  ⟨R(iθ + Φ_i) q, R(jθ + Φ_j) k⟩ = ⟨q, R((j - i)θ + (Φ_j - Φ_i)) k⟩.
-/
theorem rope_relative_rotation_invariance
    (θ phi_i phi_j i j : ℝ) (q k : Vec2) :
    dot2 (rotate2 (i * θ + phi_i) q) (rotate2 (j * θ + phi_j) k) =
    dot2 q (rotate2 ((j - i) * θ + (phi_j - phi_i)) k) := by
  rw [dot2_rotate2_eq]
  have h_ang : (j * θ + phi_j) - (i * θ + phi_i) = (j - i) * θ + (phi_j - phi_i) := by ring
  rw [h_ang]

/--
  Lemma: Antipodal Rotation Inversion
  R(θ - π) = -R(θ)
-/
theorem dot2_rotate2_sub_pi (θ : ℝ) (q k : Vec2) :
    dot2 q (rotate2 (θ - Real.pi) k) = - dot2 q (rotate2 θ k) := by
  unfold dot2 rotate2
  rw [Real.cos_sub, Real.sin_sub, Real.cos_pi, Real.sin_pi]
  ring

/--
  Lemma: 2π Full Loop Rotation Invariance
  R(θ - 2π) = R(θ)
-/
theorem dot2_rotate2_sub_two_pi (θ : ℝ) (q k : Vec2) :
    dot2 q (rotate2 (θ - 2 * Real.pi) k) = dot2 q (rotate2 θ k) := by
  unfold dot2 rotate2
  rw [Real.cos_sub, Real.sin_sub, Real.cos_two_pi, Real.sin_two_pi]
  ring

/--
  Theorem 1: Antipodal Bilinear Cancellation (Destructive Interference)
  When the accumulated gauge phase reaches an antipodal half-twist (Φ_i - Φ_j = π),
  the rotary bilinear inner product is strictly negated relative to standard RoPE:
  ⟨q̃_i, k̃_j⟩ = - ⟨q_i, k_j⟩_RoPE.
-/
theorem antipodal_bilinear_cancellation
    (θ phi_i phi_j i j : ℝ) (q k : Vec2)
    (h_twist : phi_i - phi_j = Real.pi) :
    dot2 (rotate2 (i * θ + phi_i) q) (rotate2 (j * θ + phi_j) k) =
    - dot2 (rotate2 (i * θ) q) (rotate2 (j * θ) k) := by
  rw [rope_relative_rotation_invariance]
  have h_rel : (j - i) * θ + (phi_j - phi_i) = (j - i) * θ - Real.pi := by
    linarith
  rw [h_rel, dot2_rotate2_sub_pi]
  have h_std : dot2 (rotate2 (i * θ) q) (rotate2 (j * θ) k) =
      dot2 q (rotate2 ((j - i) * θ) k) := by
    have h_inst := rope_relative_rotation_invariance θ 0 0 i j q k
    simp only [add_zero, sub_self] at h_inst
    exact h_inst
  rw [h_std]

/--
  Theorem 1b: Double Negation Bilinear Restoration (2π Cyclic Parity Return)
  When two successive inversions accumulate a 2π holonomy loop (Φ_i - Φ_j = 2π),
  the rotary bilinear inner product strictly returns to standard RoPE attention:
  ⟨q̃_i, k̃_j⟩ = + ⟨q_i, k_j⟩_RoPE.
-/
theorem double_negation_bilinear_restoration
    (θ phi_i phi_j i j : ℝ) (q k : Vec2)
    (h_loop : phi_i - phi_j = 2 * Real.pi) :
    dot2 (rotate2 (i * θ + phi_i) q) (rotate2 (j * θ + phi_j) k) =
    dot2 (rotate2 (i * θ) q) (rotate2 (j * θ) k) := by
  rw [rope_relative_rotation_invariance]
  have h_rel : (j - i) * θ + (phi_j - phi_i) = (j - i) * θ - 2 * Real.pi := by
    linarith
  rw [h_rel, dot2_rotate2_sub_two_pi]
  have h_std : dot2 (rotate2 (i * θ) q) (rotate2 (j * θ) k) =
      dot2 q (rotate2 ((j - i) * θ) k) := by
    have h_inst := rope_relative_rotation_invariance θ 0 0 i j q k
    simp only [add_zero, sub_self] at h_inst
    exact h_inst
  rw [h_std]

/-
  =============================================================================
  Section 3: Softmax Odds Ratio Attenuation & Foundations
  =============================================================================
-/

/--
  Theorem 3: Pairwise Target-to-Decoy Odds Ratio Attenuation.
  For arbitrary background partition sums Z₁ > 0 and Z₂ > 0,
  the pairwise odds ratio between decoy (logit a inverted to -a) and target (logit b)
  attenuates by exactly exp(-2 * a), cancelling all partition functions unconditionally.
-/
theorem pairwise_odds_ratio_attenuation
    (a b : ℝ) (Z₁ Z₂ : ℝ) (hZ₁ : 0 < Z₁) (hZ₂ : 0 < Z₂) :
    let p_decoy₁ := Real.exp a / Z₁
    let p_target₁ := Real.exp b / Z₁
    let p_decoy₂ := Real.exp (-a) / Z₂
    let p_target₂ := Real.exp b / Z₂
    let odds₁ := p_decoy₁ / p_target₁
    let odds₂ := p_decoy₂ / p_target₂
    odds₂ / odds₁ = Real.exp (-2 * a) := by
  intro p_decoy₁ p_target₁ p_decoy₂ p_target₂ odds₁ odds₂
  dsimp [odds₁, odds₂, p_decoy₁, p_target₁, p_decoy₂, p_target₂]
  have _h_exp_b : Real.exp b ≠ 0 := Real.exp_ne_zero b
  have _h_exp_a : Real.exp a ≠ 0 := Real.exp_ne_zero a
  have _hZ₁_ne : Z₁ ≠ 0 := ne_of_gt hZ₁
  have _hZ₂_ne : Z₂ ≠ 0 := ne_of_gt hZ₂
  rw [show (Real.exp (-a) / Z₂ / (Real.exp b / Z₂)) / (Real.exp a / Z₁ / (Real.exp b / Z₁)) =
      Real.exp (-a) / Real.exp a by field_simp]
  rw [← Real.exp_sub]
  exact congr_arg Real.exp (by ring)

/--
  Theorem 3 (Single-token formulation): Relative Odds Ratio Attenuation.
  For any background partition function Z > 0, the single-token odds ratio
  under logit inversion a ↦ -a attenuates strictly by exp(-2 * a).
-/
theorem odds_ratio_attenuation (a : ℝ) (Z : ℝ) (hZ : 0 < Z) :
    let p_std := Real.exp a / (Real.exp a + Z)
    let p_mob := Real.exp (-a) / (Real.exp (-a) + Z)
    let odds_std := p_std / (1 - p_std)
    let odds_mob := p_mob / (1 - p_mob)
    odds_mob / odds_std = Real.exp (-2 * a) := by
  intro p_std p_mob odds_std odds_mob
  dsimp [odds_std, odds_mob, p_std, p_mob]
  have h_exp_a_pos : 0 < Real.exp a := Real.exp_pos a
  have h_exp_neg_a_pos : 0 < Real.exp (-a) := Real.exp_pos (-a)
  have h_den1_pos : 0 < Real.exp a + Z := add_pos h_exp_a_pos hZ
  have h_den2_pos : 0 < Real.exp (-a) + Z := add_pos h_exp_neg_a_pos hZ
  have _h_den1_ne : Real.exp a + Z ≠ 0 := ne_of_gt h_den1_pos
  have _h_den2_ne : Real.exp (-a) + Z ≠ 0 := ne_of_gt h_den2_pos
  have _h_exp_a_ne : Real.exp a ≠ 0 := Real.exp_ne_zero a
  have hZ_ne : Z ≠ 0 := ne_of_gt hZ
  rw [show (Real.exp (-a) / (Real.exp (-a) + Z) / (1 - Real.exp (-a) / (Real.exp (-a) + Z))) /
           (Real.exp a / (Real.exp a + Z) / (1 - Real.exp a / (Real.exp a + Z))) =
      Real.exp (-a) / Real.exp a by
        field_simp
        have h_num : Real.exp a + Z - Real.exp a = Z := by ring
        have h_den : Real.exp (-a) + Z - Real.exp (-a) = Z := by ring
        rw [h_num, h_den, div_self hZ_ne]]
  rw [← Real.exp_sub]
  exact congr_arg Real.exp (by ring)

/--
  Theorem 3c (Sign Condition for Decoy Suppression):
  Decoy odds ratio strictly attenuates (exp(-2 * a) < 1) if and only if
  the initial baseline alignment is strictly positive (a > 0).
  Conversely, when a < 0, exp(-2 * a) > 1, explaining carrier-wave inversion.
-/
theorem decoy_odds_attenuates_iff_pos (a : ℝ) :
    Real.exp (-2 * a) < 1 ↔ 0 < a := by
  rw [Real.exp_lt_one_iff]
  constructor
  · intro h; nlinarith
  · intro h; nlinarith

/--
  Theorem 3d (Carrier-Wave Amplification under Negative Alignment):
  When the initial baseline RoPE alignment is negative (a < 0),
  the unconditioned half-twist amplifies the decoy odds ratio (1 < exp(-2 * a)).
-/
theorem decoy_odds_amplifies_iff_neg (a : ℝ) :
    1 < Real.exp (-2 * a) ↔ a < 0 := by
  rw [Real.one_lt_exp_iff]
  constructor
  · intro h; nlinarith
  · intro h; nlinarith

/-
  =============================================================================
  Section 4: Program Equivalence Theorem (Two-Stage Associative Scan)
  Deeply reuses Mathlib.Algebra.Group.Defs (Semigroup, Monoid)
  =============================================================================
-/

/-- 1D Affine transformation operator: f(x) = a * x + b -/
@[ext]
structure AffineStep where
  a : ℝ
  b : ℝ

/-- Composition of affine operators: (a2, b2) ∘ (a1, b1) = (a2 * a1, a2 * b1 + b2) -/
def AffineStep.comp (op2 op1 : AffineStep) : AffineStep :=
  ⟨op2.a * op1.a, op2.a * op1.b + op2.b⟩

/-- Identity affine operator: f(x) = 1 * x + 0 -/
def AffineStep.id : AffineStep :=
  ⟨1, 0⟩

instance : Mul AffineStep where
  mul := AffineStep.comp

instance : One AffineStep where
  one := AffineStep.id

/-- Associativity of affine operator composition (Semigroup law) -/
theorem affine_comp_assoc (op3 op2 op1 : AffineStep) :
    (op3 * op2) * op1 = op3 * (op2 * op1) := by
  ext
  · change (op3.a * op2.a) * op1.a = op3.a * (op2.a * op1.a)
    ring
  · change (op3.a * op2.a) * op1.b + (op3.a * op2.b + op3.b) =
      op3.a * (op2.a * op1.b + op2.b) + op3.b
    ring

instance : Semigroup AffineStep where
  mul_assoc := affine_comp_assoc

instance : Monoid AffineStep where
  one_mul := by
    intro op
    ext
    · change 1 * op.a = op.a
      ring
    · change 1 * op.b + 0 = op.b
      ring
  mul_one := by
    intro op
    ext
    · change op.a * 1 = op.a
      ring
    · change op.a * 0 + op.b = op.b
      ring

/-- Action of affine operator on scalar: op.apply x = a * x + b -/
def AffineStep.apply (op : AffineStep) (x : ℝ) : ℝ :=
  op.a * x + op.b

/-- Homomorphism property: composition acts via sequential evaluation -/
theorem affine_comp_apply (op2 op1 : AffineStep) (x : ℝ) :
    (op2 * op1).apply x = op2.apply (op1.apply x) := by
  change (op2.a * op1.a) * x + (op2.a * op1.b + op2.b) =
    op2.a * (op1.a * x + op1.b) + op2.b
  ring

/-- Sequential EMA recurrence (token-by-token autoregressive decoding):
    ω_0 = b_0, ω_{n+1} = α * ω_n + b_{n+1} -/
def ema_recurrence (α : ℝ) (b : ℕ → ℝ) : ℕ → ℝ
  | 0 => b 0
  | n + 1 => α * ema_recurrence α b n + b (n + 1)

/-- Parallel associative scan fold over affine semigroup:
    f_0 = (1, b_0), f_{n+1} = (α, b_{n+1}) * f_n -/
def affine_scan_fold (α : ℝ) (b : ℕ → ℝ) : ℕ → AffineStep
  | 0 => ⟨1, b 0⟩
  | n + 1 => ⟨α, b (n + 1)⟩ * (affine_scan_fold α b n)

/-- Stage 1 Equivalence: Affine scan evaluation at 0 equals sequential EMA recurrence -/
theorem ema_scan_equiv_recurrence (α : ℝ) (b : ℕ → ℝ) (n : ℕ) :
    (affine_scan_fold α b n).apply 0 = ema_recurrence α b n := by
  induction n with
  | zero =>
    dsimp [affine_scan_fold, ema_recurrence, AffineStep.apply]
    ring
  | succ n ih =>
    dsimp [affine_scan_fold, ema_recurrence]
    rw [affine_comp_apply]
    rw [ih]
    dsimp [AffineStep.apply]

/-- Stage 2: Parallel additive prefix sum -/
def prefix_sum (f : ℕ → ℝ) : ℕ → ℝ
  | 0 => f 0
  | n + 1 => prefix_sum f n + f (n + 1)

/-- Joint causal state tracking local torsion rate ω and accumulated phase Φ -/
structure CausalState where
  omega : ℝ
  phi : ℝ

/-- Autoregressive step updating joint causal state -/
def sequential_causal_step (α : ℝ) (b_next : ℝ) (st : CausalState) : CausalState :=
  let next_omega := α * st.omega + b_next
  ⟨next_omega, st.phi + next_omega⟩

/-- Full sequential autoregressive system initialized at token 0 -/
def sequential_system (α : ℝ) (b : ℕ → ℝ) : ℕ → CausalState
  | 0 => ⟨b 0, b 0⟩
  | n + 1 => sequential_causal_step α (b (n + 1)) (sequential_system α b n)

/-- Parallel scan outputs for omega and phi -/
def omega_scan (α : ℝ) (b : ℕ → ℝ) (n : ℕ) : ℝ :=
  (affine_scan_fold α b n).apply 0

def phi_scan (α : ℝ) (b : ℕ → ℝ) (n : ℕ) : ℝ :=
  prefix_sum (fun t => (affine_scan_fold α b t).apply 0) n

/--
  Theorem 4: Program Equivalence Theorem
  The two-stage discrete associative scan (affine EMA scan + prefix sum) is
  algebraically identical to the sequential autoregressive state recurrence for all n.
-/
theorem two_stage_scan_program_equivalence (α : ℝ) (b : ℕ → ℝ) (n : ℕ) :
    (sequential_system α b n).omega = omega_scan α b n ∧
    (sequential_system α b n).phi = phi_scan α b n := by
  induction n with
  | zero =>
    dsimp [sequential_system, omega_scan, phi_scan, affine_scan_fold,
           AffineStep.apply, prefix_sum]
    constructor <;> ring
  | succ n ih =>
    have h_step : (affine_scan_fold α b (n + 1)).apply 0 =
        α * (affine_scan_fold α b n).apply 0 + b (n + 1) := by
      dsimp [affine_scan_fold]
      rw [affine_comp_apply]
      dsimp [AffineStep.apply]
    dsimp [sequential_system, sequential_causal_step]
    rw [ih.1, ih.2]
    dsimp [omega_scan, phi_scan, prefix_sum]
    exact ⟨h_step.symm, by rw [← h_step]⟩

/-
  =============================================================================
  Section 5: Spatial Aliasing Boundary Negative Theorem
  Deeply reuses Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
  =============================================================================
-/

/-- Carrier wave in RoPE is strictly negative on the interval (π/2, 3π/2) -/
theorem aliasing_boundary_carrier_negative
    {Δ θ : ℝ} (h_mid : Real.pi / 2 < Δ * θ ∧ Δ * θ < Real.pi + Real.pi / 2) :
    Real.cos (Δ * θ) < 0 :=
  Real.cos_neg_of_pi_div_two_lt_of_lt h_mid.1 h_mid.2

/--
  Theorem 5: Aliasing Boundary Negative Theorem
  When relative displacement and frequency place the carrier wave in its negative
  half-cycle ((i-j)θ_m ∈ (π/2, 3π/2)), an unconditioned π phase shift inverts the
  negative logit to positive, strictly amplifying decoy attention (Odds_mob / Odds_std > 1).
-/
theorem aliasing_boundary_decoy_amplification
    {Δ θ C : ℝ} (hC : 0 < C)
    (h_mid : Real.pi / 2 < Δ * θ ∧ Δ * θ < Real.pi + Real.pi / 2) :
    let A_base := C * Real.cos (Δ * θ)
    let A_mob := C * Real.cos (Δ * θ - Real.pi)
    A_base < 0 ∧ 0 < A_mob ∧ 1 < Real.exp (A_mob - A_base) := by
  intro A_base A_mob
  have h_cos_neg : Real.cos (Δ * θ) < 0 :=
    Real.cos_neg_of_pi_div_two_lt_of_lt h_mid.1 h_mid.2
  have h_base_neg : A_base < 0 := by
    dsimp [A_base]
    exact mul_neg_of_pos_of_neg hC h_cos_neg
  have h_cos_sub : Real.cos (Δ * θ - Real.pi) = -Real.cos (Δ * θ) :=
    Real.cos_sub_pi (Δ * θ)
  have h_mob_eq : A_mob = -A_base := by
    dsimp [A_mob, A_base]
    rw [h_cos_sub]
    ring
  have h_mob_pos : 0 < A_mob := by
    rw [h_mob_eq]
    linarith
  have h_diff_pos : 0 < A_mob - A_base := by
    linarith
  have h_odds_amp : 1 < Real.exp (A_mob - A_base) := by
    rw [Real.one_lt_exp_iff]
    exact h_diff_pos
  exact ⟨h_base_neg, h_mob_pos, h_odds_amp⟩

/-
  =============================================================================
  Section 6: Monotonic Suppression Recovery Conditions
  =============================================================================
-/

/--
  Theorem 6a: Monotonic Suppression under Dynamic Frequency Cutoff
  When evolution frequencies are bounded below the Nyquist threshold (θ_cutoff < π / (2 * L_max)),
  the carrier wave remains in its positive half-cycle for all distances Δ ≤ L_max,
  guaranteeing global monotonic suppression (Odds_mob / Odds_std < 1).
-/
theorem cutoff_frequency_monotonic_suppression
    {L_max θ_cutoff : ℝ} (hL : 0 < L_max) (_hθ_nonneg : 0 ≤ θ_cutoff)
    (h_band : θ_cutoff < Real.pi / (2 * L_max))
    {Δ θ C : ℝ} (hΔ_nonneg : 0 ≤ Δ) (hΔ_le : Δ ≤ L_max)
    (hθ_pos : 0 ≤ θ) (hθ_le : θ ≤ θ_cutoff) (hC : 0 < C) :
    let A_base := C * Real.cos (Δ * θ)
    let A_mob := C * Real.cos (Δ * θ - Real.pi)
    0 < A_base ∧ A_mob < 0 ∧ Real.exp (A_mob - A_base) < 1 := by
  intro A_base A_mob
  have h_prod_le : Δ * θ ≤ L_max * θ_cutoff := by nlinarith
  have h_bound : L_max * θ_cutoff < Real.pi / 2 := by
    have h1 : L_max * θ_cutoff < L_max * (Real.pi / (2 * L_max)) :=
      mul_lt_mul_of_pos_left h_band hL
    have h2 : L_max * (Real.pi / (2 * L_max)) = Real.pi / 2 := by
      have hL_ne : L_max ≠ 0 := ne_of_gt hL
      have : L_max * (Real.pi / (2 * L_max)) = (L_max * Real.pi) / (2 * L_max) := by
        exact mul_div_assoc' L_max Real.pi (2 * L_max)
      rw [this]
      have h_comm : L_max * Real.pi = Real.pi * L_max := mul_comm L_max Real.pi
      rw [h_comm]
      have h_denom : 2 * L_max = 2 * L_max := rfl
      rw [h_denom]
      have h_div : (Real.pi * L_max) / (2 * L_max) = (Real.pi / 2) * (L_max / L_max) := by
        exact mul_div_mul_comm Real.pi L_max 2 L_max
      rw [h_div, div_self hL_ne, mul_one]
    linarith
  have h_prod_lt : Δ * θ < Real.pi / 2 := lt_of_le_of_lt h_prod_le h_bound
  have h_mem : Δ * θ ∈ Set.Ioo (-(Real.pi / 2)) (Real.pi / 2) := by
    have h_prod_nonneg : 0 ≤ Δ * θ := by nlinarith
    have _h_pi_pos : 0 < Real.pi := Real.pi_pos
    constructor
    · linarith
    · exact h_prod_lt
  have h_cos_pos : 0 < Real.cos (Δ * θ) := Real.cos_pos_of_mem_Ioo h_mem
  have h_base_pos : 0 < A_base := mul_pos hC h_cos_pos
  have h_cos_sub : Real.cos (Δ * θ - Real.pi) = -Real.cos (Δ * θ) :=
    Real.cos_sub_pi (Δ * θ)
  have h_mob_eq : A_mob = -A_base := by
    dsimp [A_mob, A_base]; rw [h_cos_sub]; ring
  have h_mob_neg : A_mob < 0 := by rw [h_mob_eq]; linarith
  have h_diff_neg : A_mob - A_base < 0 := by linarith
  have h_odds_att : Real.exp (A_mob - A_base) < 1 := by
    rw [Real.exp_lt_one_iff]; exact h_diff_neg
  exact ⟨h_base_pos, h_mob_neg, h_odds_att⟩

/--
  Theorem 6b: Sign-Aware Gating Monotonicity
  Conditioning the inversion on positive baseline alignment (0 < A) universally
  guarantees A_gated ≤ A and relative odds attenuation factor ≤ 1.0.
-/
theorem sign_aware_gating_monotonicity (A : ℝ) :
    let eff_phase := if 0 < A then Real.pi else 0
    let A_gated := A * Real.cos eff_phase
    A_gated ≤ A ∧ Real.exp (A_gated - A) ≤ 1 := by
  intro eff_phase A_gated
  dsimp [eff_phase, A_gated]
  split_ifs with h_pos
  · have h_cos : Real.cos Real.pi = -1 := Real.cos_pi
    rw [h_cos]
    have h_le : A * (-1) ≤ A := by linarith
    have h_sub : A * (-1) - A ≤ 0 := by linarith
    constructor
    · exact h_le
    · rw [Real.exp_le_one_iff]; exact h_sub
  · have h_cos : Real.cos 0 = 1 := Real.cos_zero
    rw [h_cos, mul_one]
    constructor
    · exact le_rfl
    · rw [sub_self, Real.exp_zero]

/-
  =============================================================================
  Section 7: Epsilon-Perturbation Robustness Theorem
  Deeply reuses Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
  =============================================================================
-/

/-- Lemma: cos x > 1/2 whenever |x| < π/3 -/
lemma cos_gt_half_of_abs_lt_pi_div_three {x : ℝ} (hx : |x| < Real.pi / 3) :
    1 / 2 < Real.cos x := by
  rw [← Real.cos_abs]
  have h_abs_nonneg : 0 ≤ |x| := abs_nonneg x
  have h_pi_three_le_pi : Real.pi / 3 ≤ Real.pi := by
    have : 0 < Real.pi := Real.pi_pos; linarith
  have h_lt := Real.cos_lt_cos_of_nonneg_of_le_pi h_abs_nonneg h_pi_three_le_pi hx
  rw [Real.cos_pi_div_three] at h_lt
  exact h_lt

/--
  Theorem 7: Epsilon-Perturbation Robustness Theorem
  Under open-loop phase drift |δ| < π/6 (±30°), the decoy suppression margin
  strictly satisfies A_base - A_mob > 0.5 * C > 0, ensuring that the suppression
  ratio exp(A_base - A_mob) strictly exceeds 1.0 without requiring exact 180° calibration.
-/
theorem epsilon_perturbation_robustness
    {C θ δ : ℝ} (hC : 0 < C)
    (hθ_nonneg : 0 ≤ θ) (hθ_le : θ ≤ Real.pi / 6)
    (hδ : |δ| < Real.pi / 6) :
    let A_base := C * Real.cos θ
    let A_mob := C * Real.cos (θ - (Real.pi + δ))
    let suppression_margin := A_base - A_mob
    0.5 * C < suppression_margin ∧ 1 < Real.exp suppression_margin := by
  intro A_base A_mob suppression_margin
  have h_cos_sub : Real.cos (θ - (Real.pi + δ)) = -Real.cos (θ - δ) := by
    have h_ang : θ - (Real.pi + δ) = (θ - δ) - Real.pi := by ring
    rw [h_ang, Real.cos_sub_pi]
  have h_diff : suppression_margin = C * (Real.cos θ + Real.cos (θ - δ)) := by
    dsimp [suppression_margin, A_base, A_mob]; rw [h_cos_sub]; ring
  have h_abs_delta : -(Real.pi / 6) < δ ∧ δ < Real.pi / 6 := abs_lt.mp hδ
  have h3 : -(Real.pi / 3) < θ - δ := by
    have _h_pi_pos : 0 < Real.pi := Real.pi_pos
    linarith [hθ_nonneg, h_abs_delta.2]
  have h_upper : θ - δ < Real.pi / 3 := by
    have _h_pi_pos : 0 < Real.pi := Real.pi_pos
    linarith [hθ_le, h_abs_delta.1]
  have h_ang_bound : |θ - δ| < Real.pi / 3 := by
    rw [abs_lt]
    exact ⟨h3, h_upper⟩
  have h_cos_rel_gt : 1 / 2 < Real.cos (θ - δ) :=
    cos_gt_half_of_abs_lt_pi_div_three h_ang_bound
  have h_mem_theta : θ ∈ Set.Ioo (-(Real.pi / 2)) (Real.pi / 2) := by
    have _h_pi_pos : 0 < Real.pi := Real.pi_pos
    constructor <;> linarith
  have h_cos_theta_pos : 0 < Real.cos θ := Real.cos_pos_of_mem_Ioo h_mem_theta
  have h_sum_gt : 1 / 2 < Real.cos θ + Real.cos (θ - δ) := by linarith
  have h_margin_gt : 0.5 * C < suppression_margin := by
    rw [h_diff]
    have h_half_c : (1 / 2 : ℝ) * C < (Real.cos θ + Real.cos (θ - δ)) * C :=
      mul_lt_mul_of_pos_right h_sum_gt hC
    have h_comm : (Real.cos θ + Real.cos (θ - δ)) * C =
        C * (Real.cos θ + Real.cos (θ - δ)) := by ring
    linarith
  have h_supp_pos : 0 < suppression_margin := by
    have : 0 < 0.5 * C := mul_pos (by norm_num) hC
    linarith
  have h_exp_gt : 1 < Real.exp suppression_margin := by
    rw [Real.one_lt_exp_iff]
    exact h_supp_pos
  exact ⟨h_margin_gt, h_exp_gt⟩

/-
  =============================================================================
  Section 8: Multi-Turn Parity-Alternating Partition Function Contraction
  Real-algebra core of paper Theorem 7 (thm:multi_turn_contraction) and
  Corollary 8 (cor:k_turn_acr). Floating-point semantics remain the
  responsibility of Proposition 6; everything below is exact real algebra.
  Deeply reuses Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
  =============================================================================
-/

/--
  Lemma (Theorem 7 (i), parity core): the n-th nominal reversal phase nπ
  contributes cos(nπ) = (-1)^n, i.e. the sign alternates with the parity of n.
-/
theorem multi_turn_parity_cos_phase (n : ℕ) :
    Real.cos (n * Real.pi) = (-1 : ℝ) ^ n :=
  Real.cos_nat_mul_pi n

/--
  Theorem 7 (i): Multi-Turn Parity Sign Flip.
  Rotating a 2D rotary vector by an extra nπ flips the rotary inner product
  sign exactly with the parity of n: even n revives the term in phase, odd n
  voids it anti-phase.
-/
theorem multi_turn_parity_sign_flip (n : ℕ) (θ : ℝ) (q k : Vec2) :
    dot2 (rotate2 (θ + n * Real.pi) q) k = (-1 : ℝ) ^ n * dot2 (rotate2 θ q) k := by
  have h_cos : Real.cos (θ + n * Real.pi) = (-1 : ℝ) ^ n * Real.cos θ :=
    Real.cos_add_nat_mul_pi θ n
  have h_sin : Real.sin (θ + n * Real.pi) = (-1 : ℝ) ^ n * Real.sin θ :=
    Real.sin_add_nat_mul_pi θ n
  unfold dot2 rotate2
  rw [h_cos, h_sin]
  ring

/--
  Theorem 7 (i): Voided-Term Identity.
  A voided anti-phase contribution satisfies
  exp(A_std - 2 * A_mob) = exp(A_std) - exp(A_std) * (1 - exp(-2 * A_mob)),
  the term-by-term form of the contraction law Z̃ = Z - ΔZ_tot.
-/
theorem multi_turn_voided_term_identity (a m : ℝ) :
    Real.exp (a - 2 * m) = Real.exp a - Real.exp a * (1 - Real.exp (-2 * m)) := by
  have h_split : Real.exp (a - 2 * m) = Real.exp a * Real.exp (-2 * m) := by
    rw [show a - 2 * m = a + (-2 * m) from by ring, Real.exp_add]
  rw [h_split]
  ring

/--
  Theorem 7 (i): Multi-Turn Partition Contraction (single voided term).
  Every voided contribution exp(a) * (1 - exp(-2 * m)) is strictly positive
  whenever the coherence alignment satisfies A_mob = m > 0.
-/
theorem multi_turn_partition_contraction (a m : ℝ) (hm : 0 < m) :
    0 < Real.exp a * (1 - Real.exp (-2 * m)) := by
  have h_exp_pos : 0 < Real.exp a := Real.exp_pos a
  have h_arg_neg : -2 * m < 0 := by linarith
  have h_exp_lt : Real.exp (-2 * m) < 1 := Real.exp_lt_one_iff.mpr h_arg_neg
  have h_one_sub_pos : 0 < 1 - Real.exp (-2 * m) := by linarith
  exact mul_pos h_exp_pos h_one_sub_pos

/--
  Theorem 7 (i): Multi-Turn Partition Contraction (summed form).
  With strictly positive voided contributions over a nonempty segment index
  set, the total ΔZ_tot > 0 and the contracted partition function obeys
  Z̃ = Z - ΔZ_tot < Z strictly.
-/
theorem multi_turn_partition_contraction_sum {ι : Type*} (s : Finset ι) (A Am : ι → ℝ)
    (Z : ℝ) (hs : s.Nonempty) (hpos : ∀ i ∈ s, 0 < Am i) :
    0 < s.sum (fun i => Real.exp (A i) * (1 - Real.exp (-2 * Am i))) ∧
      Z - s.sum (fun i => Real.exp (A i) * (1 - Real.exp (-2 * Am i))) < Z := by
  have h_term : ∀ i ∈ s, 0 < Real.exp (A i) * (1 - Real.exp (-2 * Am i)) := by
    intro i hi
    exact multi_turn_partition_contraction (A i) (Am i) (hpos i hi)
  have h_sum : 0 < s.sum (fun i => Real.exp (A i) * (1 - Real.exp (-2 * Am i))) :=
    Finset.sum_pos h_term hs
  exact ⟨h_sum, by linarith⟩

/--
  Corollary 8 (1): Uniform In-Phase Amplification Identity.
  For an in-phase token with unchanged numerator exp(a), the probability ratio
  against the contracted partition function satisfies
  exp(a) / (Z - Δ) = (exp(a) / Z) * (1 / (1 - Δ / Z)),
  where the second factor is the token-independent amplification
  T = 1 / (1 - ΔZ_tot / Z).
-/
theorem uniform_in_phase_amplification (a Z Δ : ℝ) (hZ : 0 < Z) (_hΔ : 0 < Δ) (hlt : Δ < Z) :
    Real.exp a / (Z - Δ) = (Real.exp a / Z) * (1 / (1 - Δ / Z)) := by
  have hZ_ne : Z ≠ 0 := ne_of_gt hZ
  have hsub_pos : 0 < Z - Δ := by linarith
  have hsub_ne : Z - Δ ≠ 0 := ne_of_gt hsub_pos
  have hratio_pos : 0 < 1 - Δ / Z := by
    have h_ratio : Δ / Z < 1 := (div_lt_one hZ).2 hlt
    linarith
  have hratio_ne : 1 - Δ / Z ≠ 0 := ne_of_gt hratio_pos
  field_simp [hZ_ne, hsub_ne, hratio_ne]

/--
  Corollary 8 (1): Token-Identity-Independent Amplification Factor.
  There exists T = 1 / (1 - Δ / Z) > 1 such that for every logit a the
  uniform identity exp(a) / (Z - Δ) = (exp(a) / Z) * T holds; the factor T
  does not depend on the token identity a.
-/
theorem uniform_in_phase_amplification_factor (Z Δ : ℝ) (hZ : 0 < Z) (hΔ : 0 < Δ) (hlt : Δ < Z) :
    ∃ T : ℝ, 1 < T ∧ ∀ a : ℝ, Real.exp a / (Z - Δ) = (Real.exp a / Z) * T := by
  refine ⟨1 / (1 - Δ / Z), ?_, ?_⟩
  · have hratio_pos : 0 < 1 - Δ / Z := by
      have h_ratio : Δ / Z < 1 := (div_lt_one hZ).2 hlt
      linarith
    have hratio_lt : 1 - Δ / Z < 1 := by
      have h_ratio : 0 < Δ / Z := div_pos hΔ hZ
      linarith
    exact (one_lt_div hratio_pos).2 hratio_lt
  · intro a
    exact uniform_in_phase_amplification a Z Δ hZ hΔ hlt

/--
  Theorem 7 (ii): Perturbation Sufficient Condition.
  For bounded phase perturbation |δ| ≤ π/6, the anti-phase first-order
  contraction factor satisfies 1 + cos δ ≥ 1 + √3 / 2 ≈ 1.866, keeping the
  anti-phase first-order contraction dominant under the stated perturbation.
-/
theorem one_add_cos_lower_bound (δ : ℝ) (hδ : |δ| ≤ Real.pi / 6) :
    1 + Real.sqrt 3 / 2 ≤ 1 + Real.cos δ := by
  have h_bound : Real.sqrt 3 / 2 ≤ Real.cos δ := by
    rw [← Real.cos_abs]
    have h_nonneg : 0 ≤ |δ| := abs_nonneg δ
    have h_le_pi : Real.pi / 6 ≤ Real.pi := by
      have _h_pi_pos : 0 < Real.pi := Real.pi_pos
      linarith
    have h_cos_le : Real.cos (Real.pi / 6) ≤ Real.cos |δ| :=
      Real.cos_le_cos_of_nonneg_of_le_pi h_nonneg h_le_pi hδ
    rw [Real.cos_pi_div_six] at h_cos_le
    exact h_cos_le
  linarith

end MobiusAttention
