# Change Advisory Board Meeting Notes - November 2025

_Document code: IT-MTG-001 · Version 1 · Effective date: 2025-11-04 · Owner: IT Service Management · Classification: Internal_

## Meeting Details

Date: 2025-11-04. Chair: Marcus Lee (IT Service Management). Attendees: Nadia Karim (IT Security), Elena Ruiz (Operations, WH-DAL-02), representatives from IT Infrastructure and Applications. The Change Advisory Board (CAB) meets every Tuesday to review changes to production systems.

## Approved Changes

- CHG-2291: Upgrade the warehouse management system to version 8.2 at WH-DAL-02. Maintenance window 2025-11-16 from 22:00 to 02:00 Central Time. Rollback plan tested in staging. Scanners must be updated to the new WMS app version before the next shift, or they will show error NX-3310.
- CHG-2295: Firewall rule change to allow the new carrier label service. Low risk, approved for 2025-11-06.
- CHG-2298: Replace the core switch at WH-ATL-01. Maintenance window 2025-11-09 from 01:00 to 04:00. Site manager confirmed no outbound loads during the window.

## Deferred Changes

- CHG-2301: Migrate remaining users to the new Outlook for Windows. Deferred to January 2026 because of the change freeze and open calendar sync issues (error NX-2107).
- CHG-2304: VPN client upgrade. Deferred until the certificate renewal process is fixed, to avoid a spike in NX-4012 errors.

## Peak Season Change Freeze

The CAB confirmed the annual change freeze from November 20 to January 5. During the freeze, only emergency changes approved by the IT Director and the COO may be made to systems that support warehouse operations, transportation or customer orders.

## Incidents Review

Two incidents from October were reviewed. A conveyor control outage at WH-DAL-02 on 2025-10-14 lasted 3 hours and was caused by an untested firmware update; the vendor will now be required to submit changes through CAB. A certificate expiry on the label printing server caused NX-5002 errors at three sites for 40 minutes.

## Action Items

- Marcus: publish the freeze calendar on the intranet by 2025-11-07.
- Elena: confirm WH-DAL-02 staffing for scanner updates on 2025-11-17.
- Nadia: review the vendor firmware process with Operations.
