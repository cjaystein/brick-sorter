-- Brick Sorter: tables for each signed-in person's data.
-- Paste all of this into Supabase: SQL Editor -> New query -> Run. Safe to run again.

-- One row per part (or minifigure) in a person's inventory. quantity 0 = removed.
create table if not exists public.inventory_items (
  user_id    uuid not null default auth.uid() references auth.users (id) on delete cascade,
  item_key   text not null,
  quantity   integer not null default 0,
  data       jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now(),
  primary key (user_id, item_key)
);
create index if not exists inventory_items_updated on public.inventory_items (user_id, updated_at);

-- Everything else, one row per kind: the sets in the pile, the sorting bins, preferences.
create table if not exists public.user_state (
  user_id    uuid not null default auth.uid() references auth.users (id) on delete cascade,
  key        text not null,
  value      jsonb,
  updated_at timestamptz not null default now(),
  primary key (user_id, key)
);

-- Stamp each change with the time, so other devices can fetch only what changed.
create or replace function public.touch_updated_at() returns trigger
language plpgsql set search_path = '' as $$
begin
  new.updated_at = now();
  return new;
end $$;
drop trigger if exists inventory_items_touch on public.inventory_items;
create trigger inventory_items_touch before update on public.inventory_items
  for each row execute function public.touch_updated_at();
drop trigger if exists user_state_touch on public.user_state;
create trigger user_state_touch before update on public.user_state
  for each row execute function public.touch_updated_at();

-- Only you can see or change your own rows.
alter table public.inventory_items enable row level security;
alter table public.user_state enable row level security;
drop policy if exists "own inventory" on public.inventory_items;
create policy "own inventory" on public.inventory_items for all to authenticated
  using ((select auth.uid()) = user_id) with check ((select auth.uid()) = user_id);
drop policy if exists "own state" on public.user_state;
create policy "own state" on public.user_state for all to authenticated
  using ((select auth.uid()) = user_id) with check ((select auth.uid()) = user_id);

grant select, insert, update, delete on public.inventory_items, public.user_state to authenticated;
revoke all on public.inventory_items, public.user_state from anon;
