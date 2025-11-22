(defpackage #:scenario1)
(in-package #:scenario1)

(define-control-program start-mission-one-drone-one ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 300
    )
    :contingent t
)))

(define-control-program start-mission-one-drone-two ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 60
    )
    :contingent t
)))

(define-control-program land-drone-one ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 5)
)))

(define-control-program land-drone-two ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 5)
)))

(define-control-program optional-recovery-mission ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 600
    )
    :contingent t
)))

(define-control-program broadcast-event-drone-one ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 20)
    :contingent t
)))

(define-control-program broadcast-event-drone-two ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 20)
    :contingent t
)))


(define-control-program main ()
    (with-temporal-constraint (simple-temporal :upper-bound 2400)
    (sequence (:slack nil)
        
        (parallel (:slack t)
            (start-mission-one-drone-one)
            (start-mission-one-drone-two)
        )

        (parallel (:slack t)
            (broadcast-event-drone-one)
            (broadcast-event-drone-two)
        )

        (optional-recovery-mission)
        
        (parallel (:slack t)
            (land-drone-one)
            (land-drone-two)
        )
        
    ))
)
