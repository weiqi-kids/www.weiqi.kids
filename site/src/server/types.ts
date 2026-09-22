export interface Account {
  id: string;
  display_name: string;
  is_admin: number;
  status: 'active' | 'disabled';
}
