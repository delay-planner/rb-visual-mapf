(defpackage #:scenario1)
(in-package #:scenario1)

(define-control-program start-mission-one-drone-one ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 150
	:min-observation-delay 30
	:max-observation-delay 40
    )
    :contingent t
)))

(define-control-program start-mission-one-drone-two ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 105
	:min-observation-delay 30
	:max-observation-delay 40
    )
    :contingent t
)))

(define-control-program start-mission-two-drone-three ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 72
	:min-observation-delay 5
	:max-observation-delay 10
    )
    :contingent t
)))

(define-control-program start-mission-two-drone-four ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 75
	:min-observation-delay 5
	:max-observation-delay 10
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

(define-control-program upload-a-one ()
    (declare (primitive)
    (duration (simple :lower-bound 30 :upper-bound 40)
)))

(define-control-program upload-a-two ()
    (declare (primitive)
    (duration (simple :lower-bound 30 :upper-bound 40)
)))

(define-control-program upload-b-one ()
    (declare (primitive)
    (duration (simple :lower-bound 30 :upper-bound 40)
)))

(define-control-program upload-b-two ()
    (declare (primitive)
    (duration (simple :lower-bound 30 :upper-bound 40)
)))

(define-control-program sync-one ()
    (declare (primitive)
    (duration (simple :lower-bound 1 :upper-bound 1)
)))

(define-control-program main ()
    (with-temporal-constraint (simple-temporal :upper-bound 2400)
    (sequence (:slack t)
        
        (parallel (:slack t)
            (sequence (:slack nil) 
                (start-mission-one-drone-one)
                (upload-a-one)
            )
            (sequence (:slack nil)
                (start-mission-one-drone-two)
                (upload-a-two)
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
                (upload-b-one)
            )
            (sequence (:slack nil)
                (start-mission-two-drone-four)
                (upload-b-two)
            )
        )

        (parallel (:slack t)
            (land-drone-three)
            (land-drone-four)
        )
        
    ))
)
