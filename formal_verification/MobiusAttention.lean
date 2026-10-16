/-
  =============================================================================
  MobiusAttention.lean: Formal Verification of Semantic Gauge Möbius Attention
  -----------------------------------------------------------------------------
  Dual-Licensed under Apache-2.0 and MulanPSL-2.0.
  
  This Lean 4 formalization rigorously proves the algebraic invariants,
  bilinear rotary dual modulation, topological destructive interference,
  double negation restoration, and relative odds ratio attenuation of
  the Semantic Gauge Möbius Attention (SG-Möbius-RoPE) mechanism.
  =============================================================================
-/

import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Exp

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
  Section 1: Topological Path-Integral & Flatness Invariants
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
  Section 3: Softmax Odds Ratio Attenuation
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
  have h_exp_b : Real.exp b ≠ 0 := Real.exp_ne_zero b
  have h_exp_a : Real.exp a ≠ 0 := Real.exp_ne_zero a
  have hZ₁_ne : Z₁ ≠ 0 := ne_of_gt hZ₁
  have hZ₂_ne : Z₂ ≠ 0 := ne_of_gt hZ₂
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
  have h_den1_ne : Real.exp a + Z ≠ 0 := ne_of_gt h_den1_pos
  have h_den2_ne : Real.exp (-a) + Z ≠ 0 := ne_of_gt h_den2_pos
  have h_exp_a_ne : Real.exp a ≠ 0 := Real.exp_ne_zero a
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
  · intro h
    nlinarith
  · intro h
    nlinarith

/--
  Theorem 3d (Carrier-Wave Amplification under Negative Alignment):
  When the initial baseline RoPE alignment is negative (a < 0),
  the unconditioned half-twist amplifies the decoy odds ratio (1 < exp(-2 * a)).
-/
theorem decoy_odds_amplifies_iff_neg (a : ℝ) :
    1 < Real.exp (-2 * a) ↔ a < 0 := by
  rw [Real.one_lt_exp_iff]
  constructor
  · intro h
    nlinarith
  · intro h
    nlinarith

end MobiusAttention
