import { Notice } from "@/components/Shell";

export const metadata = { title: "Not found · TGF Catalog" };

export default function NotFound() {
  return <Notice title="No page here">The address does not match a page of the catalog.</Notice>;
}
