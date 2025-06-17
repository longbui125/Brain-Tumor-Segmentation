from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau #type: ignore

def get_callbacks(
    checkpoint_path='best_model.h5',
    monitor='val_loss',
    patience_es=12,
    patience_lr=5,
    min_lr=1e-7,
    verbose=1
):
    
    early_stop = EarlyStopping(
        monitor=monitor,
        patience=patience_es,
        verbose=verbose,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor=monitor,
        factor=0.5,
        patience=patience_lr,
        min_lr=min_lr,
        verbose=verbose
    )

    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=monitor,
        save_best_only=True,
        save_weights_only=False,
        verbose=verbose
    )

    return [early_stop, reduce_lr, checkpoint]