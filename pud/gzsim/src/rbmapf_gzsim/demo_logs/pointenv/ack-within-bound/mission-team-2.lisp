(defpackage #:scenario1)
(in-package #:scenario1)

(define-control-program start-mission-one-drone-one ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 600
	:min-observation-delay 30
	:max-observation-delay 40
    )
    :contingent t
)))

(define-control-program start-mission-one-drone-two ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 600
	:min-observation-delay 30
	:max-observation-delay 40
    )
    :contingent t
)))

(define-control-program start-mission-two-drone-three ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 600
    )
    :contingent t
)))

(define-control-program start-mission-two-drone-four ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 600
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

(define-control-program land-drone-three ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 5)
)))

(define-control-program land-drone-four ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 5)
)))

(define-control-program upload-a-drone-one ()
    (declare (primitive)
    (duration (simple :lower-bound 30 :upper-bound 40)
)))

(define-control-program upload-a-drone-two ()
    (declare (primitive)
    (duration (simple :lower-bound 30 :upper-bound 40)
)))

(define-control-program upload-b-drone-three ()
    (declare (primitive)
    (duration (simple :lower-bound 30 :upper-bound 40)
)))

(define-control-program upload-b-drone-four ()
    (declare (primitive)
    (duration (simple :lower-bound 30 :upper-bound 40)
)))

(define-control-program sync-one ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 1)
)))

(define-control-program main ()
    (with-temporal-constraint (simple-temporal :upper-bound 2400)
    (sequence (:slack nil)
        
        (parallel (:slack t)
            (sequence (:slack nil) 
                (start-mission-one-drone-one)
                (upload-a-drone-one)
            )
            (sequence (:slack nil)
                (start-mission-one-drone-two)
                (upload-a-drone-two)
            )
        )

        (sync-one)

        (parallel (:slack t)
            (land-drone-one)
            (land-drone-two)
        )

        (parallel (:slack t)
            (sequence (:slack nil)
                (start-mission-two-drone-three)
                (upload-b-drone-three)
            )
            (sequence (:slack nil)
                (start-mission-two-drone-four)
                (upload-b-drone-four)
            )
        )

        (parallel (:slack t)
            (land-drone-three)
            (land-drone-four)
        )
        
    ))
)
