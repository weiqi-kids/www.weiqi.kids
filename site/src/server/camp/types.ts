export interface Showcase {
  id: string; slug: string; owner_id: string; instructor_name: string;
  title: string; summary: string; demo_links: string; before_after: string; learner_outcome: string;
  audience: string; prerequisites: string; instructor_bio: string; method: string;
  sample_problem: string; sample_cause: string; sample_solutions: string; sample_flow: string;
  setup_requirements: string; repo_url: string; license: string; allow_derivative: number;
  price: number; teaching_format: 'online' | 'onsite'; location: string; schedule_mode: 'slots' | 'vote' | 'fixed';
  fixed_start: string | null; teaching_minutes: number; group_rule: 'paid' | 'signup';
  min_size: number; max_size: number | null; pay_days: number; vote_days: number; forum_open: number;
  ttqs_needs: string; ttqs_goals: string; ttqs_outline: string; ttqs_hours: string; ttqs_methods: string; ttqs_evaluation: string; ttqs_expected: string;
  status: 'draft' | 'submitted' | 'changes_requested' | 'published' | 'archived';
  review_note: string | null; reviewed_by: string | null; reviewed_at: string | null; published_at: string | null;
  created_at: string; updated_at: string;
}

export interface Slot { id: string; showcase_id: string; weekday: number; start_time: string; active: number }

export type CohortState = 'gathering' | 'scheduling' | 'payment' | 'confirmed' | 'running' | 'ended' | 'unfilled' | 'cancelled';
export const OPEN_STATES: CohortState[] = ['gathering', 'scheduling', 'payment', 'confirmed', 'running'];

export interface Cohort {
  id: string; showcase_id: string; slot_id: string | null; seq: number; state: CohortState;
  price: number | null; teaching_format: string | null; location: string | null; schedule_mode: string | null; group_rule: string | null;
  min_size: number | null; max_size: number | null; pay_days: number | null; vote_days: number | null; teaching_minutes: number | null;
  threshold_at: string | null; propose_by: string | null; vote_closes_at: string | null;
  teaching_start: string | null; teaching_end: string | null; meeting_url: string | null;
  pay_deadline: string | null; confirmed_at: string | null; ends_at: string | null; closed_at: string | null; close_reason: string | null;
  created_at: string;
}

export interface Enrollment {
  id: string; code: string; cohort_id: string; account_id: string; amount: number;
  status: 'awaiting_payment' | 'submitted' | 'confirmed' | 'refund_pending' | 'refunded' | 'cancelled';
  paid_on: string | null; account_last5: string | null; submitted_at: string | null;
  confirmed_by: string | null; confirmed_at: string | null; refund_reason: string | null; refunded_at: string | null; created_at: string;
}

export interface DemoLink { title: string; url: string; note: string }
export const demoLinks = (s: Pick<Showcase, 'demo_links'>): DemoLink[] => { try { return JSON.parse(s.demo_links) as DemoLink[]; } catch { return []; } };
